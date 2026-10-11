//! Which frames each keyframe chain synthesizes, and which pairs of chains are blended.
//!
//! For keyframes `k_1 < ... < k_n` in a shot `first..=last`:
//!
//! - each keyframe's own frame is synthesized once and starts the chains on both of its sides;
//! - every interval `[k_i, k_i+1]` gets two chains: pass A walks forward from `k_i`, pass B
//!   walks backward from `k_i+1`, each only to the frame before the other keyframe. The two
//!   passes are blended over the interval;
//! - a single backward chain runs from `k_1` to the first frame and a single forward chain from
//!   `k_n` to the last frame.
//!
//! That is at most `2n` chains, each limited to its interval, and every chain is independent
//! of the others once its keyframe frame exists, so all of them can run at the same time.

use std::ops::RangeInclusive;

use crate::error::{VideoError, invalid};

/// Walking direction of a chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Towards later frames.
    Forward,
    /// Towards earlier frames.
    Backward,
}

/// The two passes blended over an interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pass {
    /// Forward from the interval's first keyframe.
    A,
    /// Backward from the interval's last keyframe.
    B,
}

/// Where a chain's frames go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainOutput {
    /// Straight to the output (before the first or after the last keyframe).
    Direct,
    /// Into pass `pass` of interval `interval`, to be blended with the other pass.
    Blend { interval: usize, pass: Pass },
}

/// One keyframe chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chain {
    /// Index of the chain's keyframe in [`Plan::keyframes`].
    pub keyframe: usize,
    /// Walking direction.
    pub direction: Direction,
    /// First and last frame synthesized, in walking order (the keyframe's own frame is not
    /// part of the chain: it comes from the keyframe's synthesis).
    pub start: usize,
    pub end: usize,
    pub output: ChainOutput,
}

impl Chain {
    /// Number of frames the chain synthesizes (at least 1: empty chains are not planned).
    pub fn num_frames(&self) -> usize {
        self.start.abs_diff(self.end) + 1
    }

    /// The chain's frames in walking order.
    pub fn frames(&self) -> Box<dyn Iterator<Item = usize> + Send> {
        match self.direction {
            Direction::Forward => Box::new(self.start..=self.end),
            Direction::Backward => Box::new((self.end..=self.start).rev()),
        }
    }
}

/// An interval between two neighbouring keyframes (frame numbers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    pub start: usize,
    pub end: usize,
}

impl Interval {
    /// Blend weight of pass B at frame `t`: 0 at the first keyframe, 1 at the last.
    pub fn alpha(&self, t: usize) -> f32 {
        (t - self.start) as f32 / (self.end - self.start) as f32
    }
}

/// The work for one shot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The shot's frame range.
    pub frames: RangeInclusive<usize>,
    /// Keyframe frame numbers, ascending.
    pub keyframes: Vec<usize>,
    /// Intervals between neighbouring keyframes.
    pub intervals: Vec<Interval>,
    /// The chains, ordered so that the two passes of an interval are next to each other.
    pub chains: Vec<Chain>,
}

impl Plan {
    /// Plans a shot of frames `first..=last` with the given keyframes (any order, no
    /// duplicates, all inside the shot).
    pub fn new(first: usize, last: usize, keyframes: &[usize]) -> Result<Self, VideoError> {
        if first > last {
            return invalid(format!("empty frame range {first}-{last}"));
        }
        if keyframes.is_empty() {
            return invalid("at least one keyframe is needed");
        }
        let mut keys = keyframes.to_vec();
        keys.sort_unstable();
        if let Some(w) = keys.windows(2).find(|w| w[0] == w[1]) {
            return invalid(format!("frame {} has two keyframes", w[0]));
        }
        if let Some(&k) = keys.iter().find(|&&k| k < first || k > last) {
            return invalid(format!("keyframe {k} is outside the frame range {first}-{last}"));
        }
        let n = keys.len();
        let mut chains = Vec::new();
        if keys[0] > first {
            chains.push(Chain { keyframe: 0, direction: Direction::Backward, start: keys[0] - 1, end: first, output: ChainOutput::Direct });
        }
        let intervals: Vec<Interval> = keys.windows(2).map(|w| Interval { start: w[0], end: w[1] }).collect();
        for (i, iv) in intervals.iter().enumerate() {
            if iv.end - iv.start < 2 {
                continue; // no frame between the keyframes
            }
            chains.push(Chain {
                keyframe: i,
                direction: Direction::Forward,
                start: iv.start + 1,
                end: iv.end - 1,
                output: ChainOutput::Blend { interval: i, pass: Pass::A },
            });
            chains.push(Chain {
                keyframe: i + 1,
                direction: Direction::Backward,
                start: iv.end - 1,
                end: iv.start + 1,
                output: ChainOutput::Blend { interval: i, pass: Pass::B },
            });
        }
        if keys[n - 1] < last {
            chains.push(Chain {
                keyframe: n - 1,
                direction: Direction::Forward,
                start: keys[n - 1] + 1,
                end: last,
                output: ChainOutput::Direct,
            });
        }
        Ok(Self { frames: first..=last, keyframes: keys, intervals, chains })
    }

    /// Number of output frames.
    pub fn num_frames(&self) -> usize {
        self.frames.end() - self.frames.start() + 1
    }

    /// Number of syntheses: one per keyframe plus every chain frame.
    pub fn num_syntheses(&self) -> usize {
        self.keyframes.len() + self.chains.iter().map(Chain::num_frames).sum::<usize>()
    }

    /// Number of frames made by blending two passes (the frames strictly between neighbouring
    /// keyframes). At most this many synthesized frames wait for their partner at once.
    pub fn num_blended_frames(&self) -> usize {
        self.intervals.iter().map(|iv| iv.end - iv.start - 1).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_keyframes_cover_the_shot_once_per_pass() {
        let p = Plan::new(0, 99, &[99, 0]).unwrap();
        assert_eq!(p.keyframes, [0, 99]);
        assert_eq!(p.intervals, [Interval { start: 0, end: 99 }]);
        assert_eq!(p.chains.len(), 2);
        let a = &p.chains[0];
        assert_eq!((a.direction, a.start, a.end, a.num_frames()), (Direction::Forward, 1, 98, 98));
        let b = &p.chains[1];
        assert_eq!((b.direction, b.start, b.end, b.keyframe), (Direction::Backward, 98, 1, 1));
        assert_eq!(b.frames().take(3).collect::<Vec<_>>(), [98, 97, 96]);
        assert_eq!(p.num_syntheses(), 2 + 2 * 98);
        assert_eq!(p.num_frames(), 100);
        assert_eq!(p.intervals[0].alpha(0), 0.0);
        assert_eq!(p.intervals[0].alpha(99), 1.0);
    }

    #[test]
    fn lead_in_and_lead_out_chains() {
        let p = Plan::new(10, 40, &[15, 30, 31]).unwrap();
        let summary: Vec<_> = p.chains.iter().map(|c| (c.keyframe, c.direction, c.start, c.end, c.output)).collect();
        use ChainOutput::*;
        assert_eq!(
            summary,
            [
                (0, Direction::Backward, 14, 10, Direct),
                (0, Direction::Forward, 16, 29, Blend { interval: 0, pass: Pass::A }),
                (1, Direction::Backward, 29, 16, Blend { interval: 0, pass: Pass::B }),
                // 30 and 31 are adjacent: nothing to synthesize between them.
                (2, Direction::Forward, 32, 40, Direct),
            ]
        );
        // Every frame is produced exactly once: keyframes + direct chains + blended interiors.
        let blended = p.num_blended_frames();
        assert_eq!(blended, 14);
        let direct: usize = p.chains.iter().filter(|c| c.output == Direct).map(Chain::num_frames).sum();
        assert_eq!(p.keyframes.len() + direct + blended, p.num_frames());
    }

    #[test]
    fn single_keyframe() {
        let p = Plan::new(0, 4, &[0]).unwrap();
        assert_eq!(p.chains.len(), 1);
        assert_eq!(p.num_syntheses(), 5);
        assert_eq!(Plan::new(3, 3, &[3]).unwrap().chains.len(), 0);
    }

    #[test]
    fn rejects_bad_keyframes() {
        assert!(Plan::new(0, 9, &[]).is_err());
        assert!(Plan::new(0, 9, &[3, 3]).is_err());
        assert!(Plan::new(0, 9, &[10]).is_err());
        assert!(Plan::new(5, 4, &[5]).is_err());
    }
}
