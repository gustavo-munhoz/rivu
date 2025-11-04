use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Clone)]
pub struct Snapshot {
    pub instances_seen: u64,
    pub accuracy: f64,
    pub kappa: f64,
    pub ram_hours: f64,
    pub seconds: f64,
    pub extras: BTreeMap<String, f64>,
}

impl Snapshot {
    #[inline]
    fn fmtv(v: f64) -> String {
        if v.is_nan() {
            "NaN".into()
        } else {
            format!("{:.6}", v)
        }
    }
}

impl Display for Snapshot {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "seen={}, acc={}, kappa={}, ram_h={}, t={:.6}s",
            self.instances_seen,
            Self::fmtv(self.accuracy),
            Self::fmtv(self.kappa),
            self.ram_hours,
            self.seconds
        )?;

        const ORDER: &[&str] = &["kappa_t", "kappa_m", "precision", "recall", "f1"];

        for key in ORDER {
            if let Some(v) = self.extras.get(*key) {
                write!(f, ", {}={}", key, Self::fmtv(*v))?;
            }
        }

        for (k, v) in self.extras.iter() {
            if ORDER.iter().any(|kk| *kk == k.as_str()) {
                continue;
            }
            write!(f, ", {}={}", k, Self::fmtv(*v))?;
        }

        Ok(())
    }
}
