use crate::token::{Token, TokenKind};
use crate::error::{CompilerError, ErrorType};
use std::str::Chars;
use std::iter::Peekable;

pub struct Lexer<'a> {
    input: Peekable<Chars<'a>>,
    line: usize,
    column: usize,
    source: &'a str,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer {
            input: input.chars().peekable(),
            line: 1,
            column: 1,
            source: input,
        }
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.input.next();
        if let Some(ch) = c {
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
            let start_column = self.column;
            let token_kind = match c {
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
                '/' => {
                    self.advance();
                    TokenKind::Slash
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
                c if c.is_alphabetic() => self.tokenize_identifier_or_keyword()?,
                c if c.is_whitespace() => {
                    self.advance();
                    continue;
                }
                _ => return Err(self.error(format!("Unexpected character: {}", c))),
            };

            let lexeme = self.source[start_column-1..self.column-1].to_string();
            tokens.push(Token::new(token_kind, self.line, start_column, lexeme));
        }

        Ok(tokens)
    }

    fn error(&self, message: String) -> CompilerError {
        let line_content = self.source.lines().nth(self.line - 1).unwrap_or("").to_string();
        CompilerError::new(message, self.line, self.column, line_content, ErrorType::Lexical)
    }

    // Helper methods for tokenizing specific types
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

    fn tokenize_identifier_or_keyword(&mut self) -> Result<TokenKind, CompilerError> {
        let mut ident = String::new();
        while let Some(&c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                ident.push(c);
                self.advance();
            } else {
                break;
            }
        }

        Ok(match ident.as_str() {
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
            _ => TokenKind::Identifier(ident),
        })
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::Token;

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
            Token::new(TokenKind::Identifier("User".to_string()), 1, 1, "User".to_string()),
            Token::new(TokenKind::Equal, 1, 5, "=".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 7, "{".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 1, 9, "name".to_string()),
            Token::new(TokenKind::Colon, 1, 13, ":".to_string()),
            Token::new(TokenKind::Identifier("Some".to_string()), 1, 15, "Some".to_string()),
            Token::new(TokenKind::LessThan, 1, 19, "<".to_string()),
            Token::new(TokenKind::Identifier("String".to_string()), 1, 20, "String".to_string()),
            Token::new(TokenKind::GreaterThan, 1, 25, ">".to_string()),
            Token::new(TokenKind::Comma, 1, 27, ",".to_string()),
            Token::new(TokenKind::Identifier("age".to_string()), 1, 29, "age".to_string()),
            Token::new(TokenKind::Colon, 1, 33, ":".to_string()),
            Token::new(TokenKind::Identifier("Int".to_string()), 1, 35, "Int".to_string()),
            Token::new(TokenKind::Equal, 1, 38, "=".to_string()),
            Token::new(TokenKind::IntLiteral(0), 1, 40, "0".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 42, "}".to_string()),
        ];

        assert_eq!(tokens, expected_tokens);
    }

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
            Token::new(TokenKind::ShiftRight, 1, 23, ">>".to_string()),
            Token::new(TokenKind::Not, 1, 25, "!".to_string()),
        ];

        assert_eq!(tokens, expected_tokens);
    }

    #[test]
    fn test_user_syntax() {
        let input = r#"
            User(name: Some<String>) User {
                match(name) {
                    Some(name) => {
                        name = some(name),
                        age = 0
                    },
                    None => {
                        name = none(),
                        age = 0
                    }
                }
            }
        "#;

        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();

        let expected_tokens = vec![
            Token::new(TokenKind::Identifier("User".to_string()), 1, 1, "User".to_string()),
            Token::new(TokenKind::OpenParen, 1, 5, "(".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 1, 6, "name".to_string()),
            Token::new(TokenKind::Colon, 1, 10, ":".to_string()),
            Token::new(TokenKind::Identifier("Some".to_string()), 1, 12, "Some".to_string()),
            Token::new(TokenKind::LessThan, 1, 16, "<".to_string()),
            Token::new(TokenKind::Identifier("String".to_string()), 1, 17, "String".to_string()),
            Token::new(TokenKind::GreaterThan, 1, 22, ">".to_string()),
            Token::new(TokenKind::CloseParen, 1, 24, ")".to_string()),
            Token::new(TokenKind::Identifier("User".to_string()), 1, 26, "User".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 28, "{".to_string()),
            Token::new(TokenKind::Match, 1, 30, "match".to_string()),
            Token::new(TokenKind::OpenParen, 1, 35, "(".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 1, 36, "name".to_string()),
            Token::new(TokenKind::CloseParen, 1, 40, ")".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 42, "{".to_string()),
            Token::new(TokenKind::Identifier("Some".to_string()), 1, 44, "Some".to_string()),
            Token::new(TokenKind::OpenParen, 1, 48, "(".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 1, 49, "name".to_string()),
            Token::new(TokenKind::CloseParen, 1, 53, ")".to_string()),
            Token::new(TokenKind::Arrow, 1, 55, "->".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 57, "{".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 1, 59, "name".to_string()),
            Token::new(TokenKind::Equal, 1, 63, "=".to_string()),
            Token::new(TokenKind::Identifier("some".to_string()), 1, 65, "some".to_string()),
            Token::new(TokenKind::OpenParen, 1, 69, "(".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 1, 70, "name".to_string()),
            Token::new(TokenKind::CloseParen, 1, 74, ")".to_string()),
            Token::new(TokenKind::Comma, 1, 76, ",".to_string()),
            Token::new(TokenKind::Identifier("age".to_string()), 1, 78, "age".to_string()),
            Token::new(TokenKind::Equal, 1, 82, "=".to_string()),
            Token::new(TokenKind::IntLiteral(0), 1, 84, "0".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 86, "}".to_string()),
            Token::new(TokenKind::Comma, 1, 88, ",".to_string()),
            Token::new(TokenKind::Identifier("None".to_string()), 1, 90, "None".to_string()),
            Token::new(TokenKind::Arrow, 1, 92, "->".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 94, "{".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 1, 96, "name".to_string()),
            Token::new(TokenKind::Equal, 1, 100, "=".to_string()),
            Token::new(TokenKind::Identifier("none".to_string()), 1, 102, "none".to_string()),
            Token::new(TokenKind::OpenParen, 1, 106, "(".to_string()),
            Token::new(TokenKind::CloseParen, 1, 107, ")".to_string()),
            Token::new(TokenKind::Comma, 1, 109, ",".to_string()),
            Token::new(TokenKind::Identifier("age".to_string()), 1, 111, "age".to_string()),
            Token::new(TokenKind::Equal, 1, 115, "=".to_string()),
            Token::new(TokenKind::IntLiteral(0), 1, 117, "0".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 119, "}".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 121, "}".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 123, "}".to_string()),
        ];

        assert_eq!(tokens, expected_tokens);
    }
}
