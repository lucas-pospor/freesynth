//! Progress reports and cancellation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// What a render reports while it runs. Reports come from several threads.
#[derive(Clone, Debug, PartialEq)]
pub enum Progress {
    /// The render has been planned.
    Started {
        /// Output frames.
        frames: usize,
        /// Syntheses to run (keyframe frames plus chain frames).
        syntheses: usize,
        /// Keyframe chains.
        chains: usize,
        /// Chains that run at the same time.
        parallel: usize,
    },
    /// One synthesis finished.
    Synthesized {
        /// The frame synthesized.
        frame: usize,
        /// The keyframe whose chain it belongs to.
        keyframe: usize,
        /// Syntheses finished so far, out of `total`.
        done: usize,
        total: usize,
    },
    /// One output frame was handed to the sink.
    Written {
        frame: usize,
        /// Frames written so far, out of `total`.
        done: usize,
        total: usize,
    },
}

/// A cloneable flag that asks a running render to stop. The render notices it between frames
/// and returns [`VideoError::Cancelled`](crate::VideoError::Cancelled).
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Asks every render holding a clone of this token to stop.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// True once [`cancel`](Self::cancel) has been called.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_the_flag() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!b.is_cancelled());
        a.cancel();
        assert!(b.is_cancelled());
    }
}
