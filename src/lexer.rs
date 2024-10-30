use crate::token::{Token, TokenKind};
use crate::error::{CompilerError, ErrorType};
use std::str::Chars;
use std::iter::Peekable;

pub struct Lexer<'a> {
    input: Peekable<Chars<'a>>,
    line: usize,
    column: usize,
    source: &'a str,
    position: usize,  // Add position tracking
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer {
            input: input.chars().peekable(),
            line: 1,
            column: 1,
            source: input,
            position: 0,
        }
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.input.next();
        if let Some(ch) = c {
            self.position += 1;
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
        c
    }

    fn peek(&mut self) -> Option<&char> {
        self.input.peek()
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, CompilerError> {
        let mut tokens = Vec::new();

        while let Some(&c) = self.peek() {
            let start_position = self.position;
            let start_column = self.column;
            let start_line = self.line;

            let token_kind = match c {
                // Handle comments
                '/' => {
                    self.advance(); // consume first '/'
                    if let Some(&'/') = self.peek() {
                        // Line comment found, skip until newline
                        self.skip_line_comment();
                        continue;
                    } else if let Some(&'*') = self.peek() {
                        // Block comment found, skip until */
                        self.skip_block_comment()?;
                        continue;
                    } else {
                        // Just a division operator
                        TokenKind::Slash
                    }
                }
                '{' => {
                    self.advance();
                    TokenKind::OpenBrace
                }
                '}' => {
                    self.advance();
                    TokenKind::CloseBrace
                }
                '(' => {
                    self.advance();
                    TokenKind::OpenParen
                }
                ')' => {
                    self.advance();
                    TokenKind::CloseParen
                }
                '[' => {
                    self.advance();
                    TokenKind::OpenBracket
                }
                ']' => {
                    self.advance();
                    TokenKind::CloseBracket
                }
                ',' => {
                    self.advance();
                    TokenKind::Comma
                }
                ':' => {
                    self.advance();
                    TokenKind::Colon
                }
                ';' => {
                    self.advance();
                    TokenKind::Semicolon
                }
                '.' => {
                    self.advance();
                    TokenKind::Dot
                }
                '=' => {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        self.advance();
                        TokenKind::FatArrow
                    } else {
                        TokenKind::Equal
                    }
                }
                '-' => {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        self.advance();
                        TokenKind::Arrow
                    } else {
                        TokenKind::Minus
                    }
                }
                '+' => {
                    self.advance();
                    TokenKind::Plus
                }
                '*' => {
                    self.advance();
                    TokenKind::Star
                }
                '%' => {
                    self.advance();
                    TokenKind::Percent
                }
                '&' => {
                    self.advance();
                    TokenKind::BitAnd
                }
                '|' => {
                    self.advance();
                    TokenKind::BitOr
                }
                '^' => {
                    self.advance();
                    TokenKind::BitXor
                }
                '<' => {
                    self.advance();
                    if self.peek() == Some(&'<') {
                        self.advance();
                        TokenKind::ShiftLeft
                    } else {
                        TokenKind::LessThan
                    }
                }
                '>' => {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        self.advance();
                        TokenKind::ShiftRight
                    } else {
                        TokenKind::GreaterThan
                    }
                }
                '!' => {
                    self.advance();
                    TokenKind::Not
                }
                '0'..='9' => self.tokenize_number()?,
                '"' => self.tokenize_string()?,
                '\'' => self.tokenize_char()?,
                c if c.is_alphabetic() || c == '_' => self.tokenize_identifier_or_keyword()?,
                c if c.is_whitespace() => {
                    self.skip_whitespace();
                    continue;
                }
                _ => return Err(self.error(format!("Unexpected character: {}", c))),
            };

            let lexeme = &self.source[start_position..self.position];
            tokens.push(Token::new(token_kind, start_line, start_column, lexeme.to_string()));
            self.skip_whitespace();
        }

        Ok(tokens)
    }

    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.peek() {
            if !c.is_whitespace() {
                break;
            }
            self.advance();
        }
    }

    fn tokenize_identifier_or_keyword(&mut self) -> Result<TokenKind, CompilerError> {
        let start_pos = self.position;
        
        // First character is already checked to be alphabetic or underscore
        self.advance();

        // Rest can be alphanumeric or underscore
        while let Some(&c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let ident = &self.source[start_pos..self.position];
        Ok(match ident {
            "fn" => TokenKind::Fn,
            "let" => TokenKind::Let,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "match" => TokenKind::Match,
            "loop" => TokenKind::Loop,
            "return" => TokenKind::Return,
            "struct" => TokenKind::Struct,
            "None" => TokenKind::None,
            "Some" => TokenKind::Some,
            "Option" => TokenKind::Option,
            "Mut" => TokenKind::Mut,
            "true" => TokenKind::BoolLiteral(true),
            "false" => TokenKind::BoolLiteral(false),
            _ => TokenKind::Identifier(ident.to_string()),
        })
    }

    fn tokenize_number(&mut self) -> Result<TokenKind, CompilerError> {
        let mut num_str = String::new();
        while let Some(&c) = self.peek() {
            if c.is_digit(10) || c == '.' {
                num_str.push(c);
                self.advance();
            } else {
                break;
            }
        }

        if num_str.contains('.') {
            Ok(TokenKind::FloatLiteral(num_str.parse().map_err(|_| 
                self.error("Invalid float literal".to_string()))?))
        } else {
            Ok(TokenKind::IntLiteral(num_str.parse().map_err(|_| 
                self.error("Invalid integer literal".to_string()))?))
        }
    }

    fn tokenize_string(&mut self) -> Result<TokenKind, CompilerError> {
        self.advance(); // Skip opening quote
        let mut string_content = String::new();
        
        while let Some(&c) = self.peek() {
            if c == '"' {
                self.advance(); // Skip closing quote
                return Ok(TokenKind::StringLiteral(string_content));
            }
            string_content.push(c);
            self.advance();
        }
        
        Err(self.error("Unterminated string literal".to_string()))
    }

    fn tokenize_char(&mut self) -> Result<TokenKind, CompilerError> {
        self.advance(); // Skip opening quote
        let c = self.advance().ok_or_else(|| self.error("Expected character".to_string()))?;
        if self.advance() != Some('\'') {
            return Err(self.error("Expected closing quote for char literal".to_string()));
        }
        Ok(TokenKind::CharLiteral(c))
    }

    fn error(&self, message: String) -> CompilerError {
        let line_content = self.source.lines().nth(self.line - 1).unwrap_or("").to_string();
        CompilerError::new(message, self.line, self.column, line_content, ErrorType::Lexical)
    }

    fn skip_line_comment(&mut self) {
        self.advance(); // consume the second '/'
        while let Some(&c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), CompilerError> {
        self.advance(); // consume the '*'
        while let Some(&c) = self.peek() {
            self.advance();
            if c == '*' {
                if let Some(&'/') = self.peek() {
                    self.advance(); // consume the '/'
                    return Ok(());
                }
            }
        }
        Err(self.error("Unterminated block comment".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::{Token, TokenKind};

    #[test]
    fn test_operators() {
        let input = "+ - * / % & | ^ < > << >> !";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        let expected_tokens = vec![
            Token::new(TokenKind::Plus, 1, 1, "+".to_string()),
            Token::new(TokenKind::Minus, 1, 3, "-".to_string()),
            Token::new(TokenKind::Star, 1, 5, "*".to_string()),
            Token::new(TokenKind::Slash, 1, 7, "/".to_string()),
            Token::new(TokenKind::Percent, 1, 9, "%".to_string()),
            Token::new(TokenKind::BitAnd, 1, 11, "&".to_string()),
            Token::new(TokenKind::BitOr, 1, 13, "|".to_string()),
            Token::new(TokenKind::BitXor, 1, 15, "^".to_string()),
            Token::new(TokenKind::LessThan, 1, 17, "<".to_string()),
            Token::new(TokenKind::GreaterThan, 1, 19, ">".to_string()),
            Token::new(TokenKind::ShiftLeft, 1, 21, "<<".to_string()),
            Token::new(TokenKind::ShiftRight, 1, 24, ">>".to_string()),
            Token::new(TokenKind::Not, 1, 27, "!".to_string()),
        ];

        assert_eq!(tokens, expected_tokens);
    }

    #[test]
    fn test_lexer() {
        let input = r#"
            User = {
                name: Some<String>,
                age: Int = 0
            }
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        let expected_tokens = vec![
            Token::new(TokenKind::Identifier("User".to_string()), 2, 13, "User".to_string()),
            Token::new(TokenKind::Equal, 2, 18, "=".to_string()),
            Token::new(TokenKind::OpenBrace, 2, 20, "{".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 3, 17, "name".to_string()),
            Token::new(TokenKind::Colon, 3, 21, ":".to_string()),
            Token::new(TokenKind::Some, 3, 23, "Some".to_string()),
            Token::new(TokenKind::LessThan, 3, 27, "<".to_string()),
            Token::new(TokenKind::Identifier("String".to_string()), 3, 28, "String".to_string()),
            Token::new(TokenKind::GreaterThan, 3, 34, ">".to_string()),
            Token::new(TokenKind::Comma, 3, 35, ",".to_string()),
            Token::new(TokenKind::Identifier("age".to_string()), 4, 17, "age".to_string()),
            Token::new(TokenKind::Colon, 4, 20, ":".to_string()),
            Token::new(TokenKind::Identifier("Int".to_string()), 4, 22, "Int".to_string()),
            Token::new(TokenKind::Equal, 4, 26, "=".to_string()),
            Token::new(TokenKind::IntLiteral(0), 4, 28, "0".to_string()),
            Token::new(TokenKind::CloseBrace, 5, 13, "}".to_string()),
        ];

        assert_eq!(tokens, expected_tokens);
    }

    #[test]
    fn test_comments() {
        let input = r#"
            // This is a line comment
            let x: int = 42; // End of line comment
            /* This is a
               block comment */
            let y: int = 10;
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        let expected_tokens = vec![
            Token::new(TokenKind::Let, 3, 13, "let".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 3, 17, "x".to_string()),
            Token::new(TokenKind::Colon, 3, 18, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 3, 20, "int".to_string()),
            Token::new(TokenKind::Equal, 3, 24, "=".to_string()),
            Token::new(TokenKind::IntLiteral(42), 3, 26, "42".to_string()),
            Token::new(TokenKind::Semicolon, 3, 28, ";".to_string()),
            Token::new(TokenKind::Let, 6, 13, "let".to_string()),
            Token::new(TokenKind::Identifier("y".to_string()), 6, 17, "y".to_string()),
            Token::new(TokenKind::Colon, 6, 18, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 6, 20, "int".to_string()),
            Token::new(TokenKind::Equal, 6, 24, "=".to_string()),
            Token::new(TokenKind::IntLiteral(10), 6, 26, "10".to_string()),
            Token::new(TokenKind::Semicolon, 6, 28, ";".to_string()),
        ];

        assert_eq!(tokens, expected_tokens);
    }
}
