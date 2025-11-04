use crate::core::instance_header::InstanceHeader;
use crate::core::instances::{DenseInstance, Instance};
use crate::streams::stream::Stream;

use crate::streams::arff::parser::{is_comment_or_empty, parse_header, parse_instance_values};
use std::fs::File;
use std::io::{BufRead, BufReader, Error, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug)]
pub struct ArffFileStream {
    path: PathBuf,
    reader: BufReader<File>,
    header: Arc<InstanceHeader>,
    data_start_pos: u64,
    next_line: Option<String>,
    finished: bool,
}

impl Stream for ArffFileStream {
    fn header(&self) -> &InstanceHeader {
        &self.header
    }

    fn has_more_instances(&self) -> bool {
        !self.finished
    }

    fn next_instance(&mut self) -> Option<Box<dyn Instance>> {
        if self.finished {
            return None;
        }

        let line = self.next_line.take()?;
        if let Err(_) = self.fill_next_line() {
            self.finished = true;
        }

        match parse_instance_values(&self.header, &line) {
            Ok(values) => {
                let inst = DenseInstance::new(Arc::clone(&self.header), values, 1.0);
                Some(Box::new(inst) as Box<dyn Instance>)
            }
            Err(e) => {
                eprintln!("Invalid data found in line '{line}': {e}");
                self.next_instance()
            }
        }
    }

    fn restart(&mut self) -> Result<(), Error> {
        self.reader = BufReader::new(File::open(&self.path)?);
        self.reader.seek(SeekFrom::Start(self.data_start_pos))?;
        self.finished = false;
        self.next_line = None;
        self.fill_next_line()?;
        Ok(())
    }
}

impl ArffFileStream {
    pub fn new(path: PathBuf, class_index: Option<usize>) -> Result<Self, Error> {
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);

        let (header, data_start_pos) = parse_header(&mut reader, class_index)?;

        let mut stream = ArffFileStream {
            path,
            reader,
            header: Arc::new(header),
            data_start_pos,
            next_line: None,
            finished: false,
        };

        stream.fill_next_line()?;
        Ok(stream)
    }

    fn fill_next_line(&mut self) -> Result<(), Error> {
        if self.finished {
            self.next_line = None;
            return Ok(());
        }
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line)?;
            if n == 0 {
                self.finished = true;
                self.next_line = None;
                return Ok(());
            }
            if !is_comment_or_empty(&line) {
                self.next_line = Some(line.trim().to_string());
                return Ok(());
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::ErrorKind;
    use std::io::Write;
    use tempfile::{NamedTempFile, tempdir};

    fn write_arff(contents: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(contents.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn parse_header_and_first_instances() {
        let arff = r#"%
@relation weather
@attribute outlook {sunny, overcast, rainy}
@attribute temperature numeric
@attribute humidity numeric
@attribute windy {TRUE, FALSE}
@attribute play {yes, no}
@data
sunny,85,85,FALSE,no
sunny,80,90,TRUE,no
overcast,83,86,FALSE,yes
rainy,70,96,FALSE,yes
?,75,?,TRUE,yes
"#;
        let tf = write_arff(arff);
        let mut stream = ArffFileStream::new(tf.path().to_path_buf(), Some(4)).expect("open");
        let h = stream.header();
        assert_eq!(h.relation_name(), "weather");
        assert_eq!(h.number_of_attributes(), 5);

        let inst1 = stream.next_instance().expect("inst1");
        let v1 = inst1.to_vec();
        assert_eq!(v1, vec![0.0, 85.0, 85.0, 1.0, 1.0]);

        let _ = stream.next_instance().unwrap();
        let _ = stream.next_instance().unwrap();
        let _ = stream.next_instance().unwrap();

        let inst5 = stream.next_instance().unwrap();
        assert!(inst5.is_missing_at_index(0).unwrap());
        assert!(inst5.is_missing_at_index(2).unwrap());
        assert!(!stream.has_more_instances());

        stream.restart().unwrap();
        let inst1_again = stream.next_instance().unwrap();
        assert_eq!(inst1_again.to_vec(), v1);
    }

    #[test]
    fn new_missing_file_returns_err_not_found() {
        let err = ArffFileStream::new("no/such/file.arff".into(), Some(0)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn header_without_data_errors_unexpected_eof() {
        let tf = write_arff("@relation r\n@attribute a numeric\n");
        let err = ArffFileStream::new(tf.path().to_path_buf(), Some(0)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnexpectedEof);
    }

    #[test]
    fn unsupported_header_directive_errors() {
        let tf = write_arff("@relation r\n@foo bar\n@data\n1\n");
        let err = ArffFileStream::new(tf.path().to_path_buf(), Some(0)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn nominal_domain_without_closing_brace_errors() {
        let tf = write_arff("@relation r\n@attribute outlook {sunny, rainy\n@data\nsunny\n");
        let err = ArffFileStream::new(tf.path().to_path_buf(), Some(0)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn data_row_with_wrong_arity_returns_none() {
        let tf = write_arff("@relation r\n@attribute a numeric\n@attribute b numeric\n@data\n1\n");
        let mut stream = ArffFileStream::new(tf.path().to_path_buf(), Some(1)).unwrap();
        assert!(stream.next_instance().is_none());
    }

    #[test]
    fn invalid_numeric_value_returns_none() {
        let tf = write_arff("@relation r\n@attribute x numeric\n@data\nabc\n");
        let mut stream = ArffFileStream::new(tf.path().to_path_buf(), Some(0)).unwrap();
        assert!(stream.next_instance().is_none());
    }

    #[test]
    fn fill_next_line_when_finished_is_noop() {
        let tf = write_arff("@relation r\n@attribute a numeric\n@data\n1\n");
        let mut s = ArffFileStream::new(tf.path().to_path_buf(), Some(0)).unwrap();
        s.finished = true;
        s.next_line = Some("x".into());
        s.fill_next_line().unwrap();
        assert!(s.next_line.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn next_instance_sets_finished_on_io_error() {
        let tf = write_arff("@relation r\n@attribute a numeric\n@data\n1\n2\n");
        let mut s = ArffFileStream::new(tf.path().to_path_buf(), Some(0)).unwrap();
        let _ = s.next_instance().unwrap();
        let dir = tempdir().unwrap();
        s.reader = BufReader::new(File::open(dir.path()).unwrap());
        let _ = s.next_instance();
        assert!(s.finished);
    }

    #[test]
    fn parse_header_attribute_before_relation_is_seen() {
        let tf = write_arff("@attribute a numeric\n@data\n1\n");
        let s = ArffFileStream::new(tf.path().to_path_buf(), Some(0)).unwrap();
        assert_eq!(s.header().number_of_attributes(), 1);
        assert_eq!(s.header().relation_name(), "unnamed_relation");
    }

    #[test]
    #[cfg(not(windows))]
    fn restart_after_file_removed_returns_err() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.arff");
        fs::write(&path, "@relation r\n@attribute x numeric\n@data\n1\n").unwrap();
        let mut stream = ArffFileStream::new(path.clone(), Some(0)).unwrap();
        fs::remove_file(&path).unwrap();
        let err = stream.restart().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }
}
