//! A minimal command-line tokenizer: options start with one or two dashes, everything else is a
//! value.

use std::str::FromStr;

/// One command-line token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arg {
    /// An option, without its leading dashes.
    Opt(String),
    /// A bare value.
    Value(String),
}

impl Arg {
    /// The token as it was written (dashes restored as one `-`).
    pub fn text(&self) -> String {
        match self {
            Arg::Opt(n) => format!("-{n}"),
            Arg::Value(v) => v.clone(),
        }
    }
}

/// A cursor over the arguments of one command.
pub struct Args {
    items: Vec<String>,
    pos: usize,
}

/// Whether `s` is an option name rather than a value (negative numbers are values).
fn is_option(s: &str) -> bool {
    s.len() > 1 && s.starts_with('-') && !s[1..].starts_with(|c: char| c.is_ascii_digit() || c == '.')
}

impl Args {
    pub fn new(items: Vec<String>) -> Self {
        Self { items, pos: 0 }
    }

    /// The next token, or `None` at the end.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Arg> {
        let s = self.items.get(self.pos)?.clone();
        self.pos += 1;
        Some(if is_option(&s) { Arg::Opt(s.trim_start_matches('-').to_string()) } else { Arg::Value(s) })
    }

    /// The value following option `name`.
    pub fn value(&mut self, name: &str) -> Result<String, String> {
        let v = self.items.get(self.pos).ok_or(format!("missing value for -{name}"))?.clone();
        self.pos += 1;
        Ok(v)
    }

    /// The value following option `name`, parsed.
    pub fn parse<T: FromStr>(&mut self, name: &str) -> Result<T, String> {
        let v = self.value(name)?;
        v.trim().parse().map_err(|_| format!("bad value '{v}' for -{name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Args {
        Args::new(s.split_whitespace().map(String::from).collect())
    }

    #[test]
    fn tokens() {
        let mut a = args("--image x.png -weight -0.5 file --seed 3 -");
        assert_eq!(a.next(), Some(Arg::Opt("image".into())));
        assert_eq!(a.value("image").unwrap(), "x.png");
        assert_eq!(a.next(), Some(Arg::Opt("weight".into())));
        assert_eq!(a.parse::<f32>("weight").unwrap(), -0.5);
        assert_eq!(a.next(), Some(Arg::Value("file".into())));
        assert_eq!(a.next(), Some(Arg::Opt("seed".into())));
        assert!(a.parse::<u64>("seed").is_ok());
        assert_eq!(a.next(), Some(Arg::Value("-".into())));
        assert_eq!(a.next(), None);
        assert!(a.value("x").is_err());
        assert!(args("--n abc").parse::<usize>("n").is_err());
    }
}
