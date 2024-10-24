use crate::token::Token;
use std::str::Chars;
use std::iter::Peekable;

pub struct Lexer<'a> {
    input: Peekable<Chars<'a>>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let lexer = Lexer {
            input: input.chars().peekable(),
        };
        lexer
    }

    fn advance(&mut self) -> Option<char> {
        self.input.next()
    }

    fn peek(&mut self) -> Option<&char> {
        self.input.peek()
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();

        while let Some(&c) = self.peek() {
            match c {
                '{' => {
                    tokens.push(Token::OpenBrace);
                    self.advance();
                }
                '}' => {
                    tokens.push(Token::CloseBrace);
                    self.advance();
                }
                '(' => {
                    tokens.push(Token::OpenParen);
                    self.advance();
                }
                ')' => {
                    tokens.push(Token::CloseParen);
                    self.advance();
                }
                ',' => {
                    tokens.push(Token::Comma);
                    self.advance();
                }
                ':' => {
                    tokens.push(Token::Colon);
                    self.advance();
                }
                '=' => {
                    if self.peek() == Some(&'>') {
                        tokens.push(Token::Arrow);
                        self.advance();
                    } else {
                        tokens.push(Token::Equal);
                    }
                }
                '-' => {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        tokens.push(Token::Arrow);
                        self.advance();
                    } else {
                        tokens.push(Token::Minus);
                    }
                }
                '+' => {
                    tokens.push(Token::Plus);
                    self.advance();
                }
                '*' => {
                    tokens.push(Token::Star);
                    self.advance();
                }
                '/' => {
                    tokens.push(Token::Slash);
                    self.advance();
                }
                '%' => {
                    tokens.push(Token::Percent);
                    self.advance();
                }
                '&' => {
                    tokens.push(Token::BitAnd);
                    self.advance();
                }
                '|' => {
                    tokens.push(Token::BitOr);
                    self.advance();
                }
                '^' => {
                    tokens.push(Token::BitXor);
                    self.advance();
                }
                '<' => {
                    self.advance();
                    if self.peek() == Some(&'<') {
                        tokens.push(Token::ShiftLeft);
                        self.advance();
                    } else {
                        tokens.push(Token::LessThan);
                    }
                }
                '>' => {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        tokens.push(Token::ShiftRight);
                        self.advance();
                    } else {
                        tokens.push(Token::GreaterThan);
                    }
                }
                '!' => {
                    tokens.push(Token::Not);
                    self.advance();
                }
                '0'..='9' => tokens.push(self.number()),
                '"' => tokens.push(self.string()),
                '\'' => tokens.push(self.char_literal()),
                c if c.is_alphabetic() => tokens.push(self.identifier_or_keyword()),
                c if c.is_whitespace() => {
                    self.advance();
                }
                _ => panic!("Unexpected character: {}", c),
            }
        }

        tokens
    }

    fn number(&mut self) -> Token {
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
            Token::FloatLiteral(num_str.parse().unwrap())
        } else {
            Token::IntLiteral(num_str.parse().unwrap())
        }
    }

    fn string(&mut self) -> Token {
        self.advance(); // Skip the opening quote
        let mut string_content = String::new();
        while let Some(&c) = self.peek() {
            if c == '"' {
                self.advance(); // Skip the closing quote
                break;
            }
            string_content.push(c);
            self.advance();
        }
        Token::StringLiteral(string_content)
    }

    fn char_literal(&mut self) -> Token {
        self.advance(); // Skip the opening quote
        let c = self.advance().expect("Expected a character");
        if self.advance() == Some('\'') {
            Token::CharLiteral(c)
        } else {
            panic!("Expected closing quote for char literal");
        }
    }

    fn identifier_or_keyword(&mut self) -> Token {
        let mut ident = String::new();
        while let Some(&c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                ident.push(c);
                self.advance();
            } else {
                break;
            }
        }

        match ident.as_str() {
            "do" => Token::Do,
            "match" => Token::Match,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "for" => Token::For,
            "return" => Token::Return,
            "loop" => Token::Loop,
            _ => Token::Identifier(ident),
        }
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
        let tokens = lexer.tokenize();

        let expected_tokens = vec![
            Token::Identifier("User".to_string()),
            Token::Equal,
            Token::OpenBrace,
            Token::Identifier("name".to_string()),
            Token::Colon,
            Token::Identifier("Some".to_string()),
            Token::LessThan,
            Token::Identifier("String".to_string()),
            Token::GreaterThan,
            Token::Comma,
            Token::Identifier("age".to_string()),
            Token::Colon,
            Token::Identifier("Int".to_string()),
            Token::Equal,
            Token::IntLiteral(0),
            Token::CloseBrace,
        ];

        assert_eq!(tokens, expected_tokens);
    }

    #[test]
    fn test_operators() {
        let input = "+ - * / % & | ^ < > << >> !";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();

        let expected_tokens = vec![
            Token::Plus,
            Token::Minus,
            Token::Star,
            Token::Slash,
            Token::Percent,
            Token::BitAnd,
            Token::BitOr,
            Token::BitXor,
            Token::LessThan,
            Token::GreaterThan,
            Token::ShiftLeft,
            Token::ShiftRight,
            Token::Not,
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
        let tokens = lexer.tokenize();

        let expected_tokens = vec![
            Token::Identifier("User".to_string()),
            Token::OpenParen,
            Token::Identifier("name".to_string()),
            Token::Colon,
            Token::Identifier("Some".to_string()),
            Token::LessThan,
            Token::Identifier("String".to_string()),
            Token::GreaterThan,
            Token::CloseParen,
            Token::Identifier("User".to_string()),
            Token::OpenBrace,
            Token::Match,  // Updated to use Token::Match
            Token::OpenParen,
            Token::Identifier("name".to_string()),
            Token::CloseParen,
            Token::OpenBrace,
            Token::Identifier("Some".to_string()),
            Token::OpenParen,
            Token::Identifier("name".to_string()),
            Token::CloseParen,
            Token::Arrow,
            Token::OpenBrace,
            Token::Identifier("name".to_string()),
            Token::Equal,
            Token::Identifier("some".to_string()),
            Token::OpenParen,
            Token::Identifier("name".to_string()),
            Token::CloseParen,
            Token::Comma,
            Token::Identifier("age".to_string()),
            Token::Equal,
            Token::IntLiteral(0),
            Token::CloseBrace,
            Token::Comma,
            Token::Identifier("None".to_string()),
            Token::Arrow,
            Token::OpenBrace,
            Token::Identifier("name".to_string()),
            Token::Equal,
            Token::Identifier("none".to_string()),
            Token::OpenParen,
            Token::CloseParen,
            Token::Comma,
            Token::Identifier("age".to_string()),
            Token::Equal,
            Token::IntLiteral(0),
            Token::CloseBrace,
            Token::CloseBrace,
            Token::CloseBrace,
        ];

        assert_eq!(tokens, expected_tokens);
    }
}
