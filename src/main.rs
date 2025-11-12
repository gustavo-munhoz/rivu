use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;

use rivu::evaluation::{CurveFormat, Snapshot};
use rivu::tasks::PrequentialEvaluator;
use rivu::ui::cli::args::{Cli, Command};
use rivu::ui::cli::{drivers::InquireDriver, wizard::prompt_choice};
use rivu::ui::types::build::{build_evaluator, build_learner, build_stream};
use rivu::ui::types::choices::{DumpFormat, TaskChoice};

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const FG_CYAN: &str = "\x1b[36m";
const FG_GREEN: &str = "\x1b[32m";
const FG_MAGENTA: &str = "\x1b[35m";
const FG_BLUE: &str = "\x1b[34m";
const FG_GREY: &str = "\x1b[90m";

fn main() -> Result<()> {
    let cli = Cli::parse();

    let task: TaskChoice = match cli.command {
        Some(Command::Run(args)) => args.into_task_choice()?,
        None => {
            let driver = InquireDriver;
            prompt_choice::<TaskChoice, _>(&driver).context("failed while prompting for task")?
        }
    };

    let render: JoinHandle<()>;

    let dump_path: Option<PathBuf>;
    let dump_format: DumpFormat;
    let mut runner = match task {
        TaskChoice::EvaluatePrequential(p) => {
            let stream_choice = p.stream;
            let evaluator_choice = p.evaluator;
            let learner_choice = p.learner;
            let max_instances = p.max_instances;
            let max_seconds = p.max_seconds;
            let sample_freq = p.sample_frequency;
            let mem_check_freq = p.mem_check_frequency;
            dump_path = p.dump_file;
            dump_format = p.dump_format;

            let header: Vec<String> = vec![
                format!("{BOLD}{FG_CYAN}▶ Prequential Evaluation{RESET}"),
                format!(
                    "{DIM}sample_freq={}{RESET}  {DIM}mem_check_freq={}{RESET}  {}",
                    sample_freq,
                    mem_check_freq,
                    timestamp_now()
                ),
                format!(
                    "{FG_GREY}────────────────────────────────────────────────────────────────────────{RESET}"
                ),
            ];

            let stream = build_stream(stream_choice).context("failed to build stream")?;
            let evaluator =
                build_evaluator(evaluator_choice).context("failed to build evaluator")?;
            let learner = build_learner(learner_choice).context("failed to build learner")?;

            let (tx, rx) = std::sync::mpsc::channel();

            render = std::thread::spawn(move || {
                render_status_with_header(rx, header, 150, max_instances, max_seconds)
            });

            PrequentialEvaluator::new(
                learner,
                stream,
                evaluator,
                max_instances,
                max_seconds,
                sample_freq,
                mem_check_freq,
            )
            .context("failed to construct PrequentialEvaluator")?
            .with_progress(tx)
        }
    };

    runner.run().context("runner failed")?;

    if let Some(path) = dump_path
        && !path.as_os_str().is_empty()
    {
        runner
            .curve()
            .export(&path, CurveFormat::from(dump_format))
            .with_context(|| format!("failed to export snapshots to {}", path.display()))?;
    }

    drop(runner);
    let _ = render.join();

    Ok(())
}

/// Print header once, then refresh a single line with status.
/// Shows: seen, acc, κ, κₜ/κₘ (if present in `extras`), ips (throughput),
/// RAM-hours, elapsed time, and small progress bars for instances/time if limits exist.
pub fn render_status_with_header(
    rx: Receiver<Snapshot>,
    header_lines: Vec<String>,
    repaint_every_ms: u64,
    max_instances: Option<u64>,
    max_seconds: Option<u64>,
) {
    for line in &header_lines {
        println!("{line}");
    }

    println!();
    let _ = io::stdout().flush();

    let tick = Duration::from_millis(repaint_every_ms);
    let mut last_draw = Instant::now();
    let mut last_snap: Option<Snapshot> = None;
    let mut prev_for_ips: Option<Snapshot> = None;

    loop {
        match rx.recv_timeout(tick) {
            Ok(s) => {
                prev_for_ips = last_snap.clone();
                last_snap = Some(s);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                if let Some(s) = last_snap.take() {
                    print!(
                        "\r{}\x1B[K\n",
                        format_status(&s, prev_for_ips.as_ref(), max_instances, max_seconds)
                    );
                    let _ = io::stdout().flush();
                }
                break;
            }
        }

        if last_draw.elapsed() >= tick {
            if let Some(s) = last_snap.as_ref() {
                let line = format_status(s, prev_for_ips.as_ref(), max_instances, max_seconds);
                print!("\r{}\x1B[K", line);
                let _ = io::stdout().flush();
            }
            last_draw = Instant::now();
        }
    }
}

fn format_status(
    s: &Snapshot,
    _prev: Option<&Snapshot>,
    max_instances: Option<u64>,
    max_seconds: Option<u64>,
) -> String {
    let seen = s.instances_seen;
    let acc = fmtf(s.accuracy * 100.0, 12);
    let kappa = fmtf(s.kappa * 100.0, 12);

    let (mut prec, mut rec, mut f1) = (String::new(), String::new(), String::new());
    if let Some(extras) = snapshot_extras(s) {
        if let Some(v) = extras.get("precision") {
            prec = format!("  {DIM}P{RESET} {}", fmtf(*v, 6));
        }
        if let Some(v) = extras.get("recall") {
            rec = format!("  {DIM}R{RESET} {}", fmtf(*v, 6));
        }
        if let Some(v) = extras.get("f1") {
            f1 = format!("  {DIM}F1{RESET} {}", fmtf(*v, 6));
        }
    }

    let mut line = format!(
        "{FG_GREEN}{BOLD}seen{RESET} {:>9}  \
         {FG_CYAN}{BOLD}acc{RESET} {:>7}% \
         {FG_MAGENTA}{BOLD}κ{RESET} {:>7}% \
         {}{}{}  \
         {DIM}ram_h{RESET} {:>8.16e}  \
         {DIM}t{RESET} {:>7.6}s",
        seen, acc, kappa, prec, rec, f1, s.ram_hours, s.seconds
    );

    let bar_w = 15usize;
    if let Some(mi) = max_instances {
        let inst_bar = progress_bar(seen as f64, mi as f64, bar_w);
        line.push_str(&format!("  {DIM}[inst]{RESET} {}", inst_bar));
    }
    if let Some(ms) = max_seconds {
        let time_bar = progress_bar(s.seconds, ms as f64, bar_w);
        line.push_str(&format!("  {DIM}[time]{RESET} {}", time_bar));
    }

    line
}

fn snapshot_extras(s: &Snapshot) -> Option<&std::collections::BTreeMap<String, f64>> {
    Some(&s.extras)
}

fn progress_bar(current: f64, total: f64, width: usize) -> String {
    if total.is_finite() && total > 0.0 {
        let ratio = (current / total).clamp(0.0, 1.0);
        let filled = (ratio * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        return format!(
            "[{}{}] {:>3.0}%",
            "█".repeat(filled),
            "░".repeat(empty),
            ratio * 100.0
        );
    }

    String::new()
}

fn fmtf(x: f64, prec: usize) -> String {
    if x.is_nan() {
        format!("{DIM}NaN{RESET}")
    } else {
        format!("{:>1$.prec$}", x, 6, prec = prec)
    }
}
fn timestamp_now() -> String {
    use chrono::{Local, SecondsFormat};
    let now = Local::now();
    format!(
        "{DIM}{}{}",
        now.to_rfc3339_opts(SecondsFormat::Secs, true),
        RESET
    )
}
