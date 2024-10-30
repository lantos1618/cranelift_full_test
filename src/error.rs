use std::fmt;

#[derive(Debug)]
pub struct CompilerError {
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub source_line: String,
    pub error_type: ErrorType,
}

#[derive(Debug)]
pub enum ErrorType {
    Lexical,
    Syntax,
    Semantic,
    CodeGen,
}

impl ErrorType {
    fn as_str(&self) -> &str {
        match self {
            ErrorType::Lexical => "Lexer",
            ErrorType::Syntax => "Parser",
            ErrorType::Semantic => "Type Checker",
            ErrorType::CodeGen => "Code Generator",
        }
    }
}

impl CompilerError {
    pub fn new(message: String, line: usize, column: usize, source_line: String, error_type: ErrorType) -> Self {
        CompilerError {
            message,
            line,
            column,
            source_line,
            error_type,
        }
    }

    pub fn display(&self) -> String {
        let mut error_display = String::new();
        error_display.push_str(&format!("[{}] Error: {} at line {}, column {}\n", 
            self.error_type.as_str(),
            self.message, 
            self.line, 
            self.column
        ));
        error_display.push_str(&format!("{}\n", self.source_line));
        error_display.push_str(&format!("{}^\n", " ".repeat(self.column - 1)));
        error_display
    }
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.display())
    }
}

impl std::error::Error for CompilerError {}
