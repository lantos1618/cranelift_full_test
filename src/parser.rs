use std::iter::Peekable;
use std::vec::IntoIter;
use crate::token::{Token, TokenKind};  // Add TokenKind import
use crate::ast::{Expr, Stmt, Program, BinOp, AstType, FuncDecl};
use crate::error::{CompilerError, ErrorType};

pub struct Parser {
    tokens: Peekable<IntoIter<Token>>,
    source_code: String,
}

impl Parser {
    pub fn new(tokens: Vec<Token>, source_code: String) -> Self {
        Parser {
            tokens: tokens.into_iter().peekable(),
            source_code,
        }
    }

    pub fn parse(&mut self) -> Result<Program, CompilerError> {
        let mut statements = Vec::new();
        while let Some(_) = self.tokens.peek() {
            statements.push(self.parse_statement()?);
        }
        Ok(Program::new(statements))
    }

    fn parse_statement(&mut self) -> Result<Stmt, CompilerError> {
        match self.tokens.peek().map(|t| &t.kind) {
            Some(TokenKind::Fn) => {
                Ok(self.parse_function_definition()?)
            }
            Some(TokenKind::Return) => {
                self.tokens.next(); // consume Return token
                let expr = self.parse_expression()?;
                Ok(Stmt::Return(Box::new(expr)))
            }
            Some(TokenKind::Identifier(_)) => {
                if self.is_struct_definition() {
                    Ok(self.parse_struct_definition()?)
                } else if self.is_assignment() {
                    Ok(self.parse_assignment()?)
                } else {
                    let expr = self.parse_expression()?;
                    Ok(Stmt::ExprStmt(Box::new(expr)))
                }
            }
            _ => {
                let expr = self.parse_expression()?;
                Ok(Stmt::ExprStmt(Box::new(expr)))
            }
        }
    }

    fn is_assignment(&mut self) -> bool {
        if let Some(Token { kind: TokenKind::Identifier(_), .. }) = self.tokens.peek() {
            let mut iter = self.tokens.clone();
            iter.next(); // Skip identifier
            if let Some(Token { kind: TokenKind::Equal, .. }) = iter.peek() {
                return true;
            }
        }
        false
    }

    fn parse_assignment(&mut self) -> Result<Stmt, CompilerError> {
        if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
            self.tokens.next(); // Skip '='
            let expr = self.parse_expression()?;
            Ok(Stmt::VarAssign {
                name,
                expr: Box::new(expr),
            })
        } else {
            Err(self.error("Expected identifier for assignment"))
        }
    }

    fn parse_expression(&mut self) -> Result<Expr, CompilerError> {
        let left = self.parse_primary()?;
        if let Some(op) = self.parse_operator() {
            let right = self.parse_primary()?;
            Ok(Expr::BinaryOp(Box::new(left), op, Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, CompilerError> {
        match self.tokens.next().map(|t| t.kind) {
            Some(TokenKind::IntLiteral(value)) => Ok(Expr::IntLiteral(value.into())), // Convert i32 to i64
            Some(TokenKind::BoolLiteral(value)) => Ok(Expr::BoolLiteral(value)),
            Some(TokenKind::Identifier(name)) => {
                // Check if this is a function call
                if let Some(Token { kind: TokenKind::OpenParen, .. }) = self.tokens.peek() {
                    self.tokens.next(); // consume (
                    let mut args = Vec::new();
                    
                    while let Some(token) = self.tokens.peek() {
                        if token.kind == TokenKind::CloseParen {
                            self.tokens.next(); // consume )
                            break;
                        }
                        
                        args.push(self.parse_expression()?);
                        
                        if let Some(Token { kind: TokenKind::Comma, .. }) = self.tokens.peek() {
                            self.tokens.next(); // consume comma
                        }
                    }
                    
                    Ok(Expr::FuncCall(name.to_string(), args))
                } else {
                    Ok(Expr::Variable(name.to_string()))
                }
            }
            _ => Err(self.error("Unexpected token")),
        }
    }

    fn parse_operator(&mut self) -> Option<BinOp> {
        match self.tokens.peek().map(|t| &t.kind) {
            Some(TokenKind::Plus) => {
                self.tokens.next();
                Some(BinOp::Add)
            }
            Some(TokenKind::Minus) => {
                self.tokens.next();
                Some(BinOp::Subtract)
            }
            Some(TokenKind::Star) => {
                self.tokens.next();
                Some(BinOp::Multiply)
            }
            Some(TokenKind::Slash) => {
                self.tokens.next();
                Some(BinOp::Divide)
            }
            Some(TokenKind::Percent) => {
                self.tokens.next();
                Some(BinOp::Modulus)
            }
            _ => None,
        }
    }

    fn parse_function_definition(&mut self) -> Result<Stmt, CompilerError> {
        // Assume the current token is the function keyword
        self.tokens.next(); // Consume 'fn'
        
        let name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
            name
        } else {
            Err(self.error("Expected function name"))?
        };

        // Parse parameter list and return type
        let params = self.parse_parameters()?;
        let return_type = self.parse_return_type()?;

        // Create the function declaration
        let func_decl = FuncDecl {
            name,
            params,
            return_type,
        };

        // Parse function body
        let body = self.parse_block()?;

        Ok(Stmt::FuncDef {
            func_decl,
            body: Box::new(body),
        })
    }

    fn parse_parameters(&mut self) -> Result<Vec<(String, AstType)>, CompilerError> {
        let mut params = Vec::new();
        
        // Expect opening parenthesis
        self.expect_token(&TokenKind::OpenParen)?;
        
        // Parse parameters until we hit )
        while let Some(token) = self.tokens.peek() {
            if token.kind == TokenKind::CloseParen {
                self.tokens.next(); // consume the )
                break;
            }

            // Parse parameter name
            let param_name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
                name
            } else {
                Err(self.error("Expected parameter name"))?
            };

            // Expect :
            self.expect_token(&TokenKind::Colon)?;

            // Parse parameter type
            let param_type = self.parse_type()?;
            params.push((param_name, param_type));

            // Handle optional comma
            if let Some(Token { kind: TokenKind::Comma, .. }) = self.tokens.peek() {
                self.tokens.next();
            }
        }

        Ok(params)
    }

    fn parse_return_type(&mut self) -> Result<AstType, CompilerError> {
        // Check for ->
        if let Some(Token { kind: TokenKind::Arrow, .. }) = self.tokens.peek() {
            self.tokens.next();
            self.parse_type()
        } else {
            Ok(AstType::Void)
        }
    }

    fn parse_block(&mut self) -> Result<Stmt, CompilerError> {
        // Expect opening brace
        self.expect_token(&TokenKind::OpenBrace)?;
        
        let mut statements = Vec::new();
        
        // Parse statements until we hit }
        while let Some(token) = self.tokens.peek() {
            if token.kind == TokenKind::CloseBrace {  // Compare TokenKind instead of dereferencing
                self.tokens.next();
                break;
            }
            statements.push(self.parse_statement()?);
        }

        Ok(Stmt::Block(statements))
    }

    fn parse_struct_definition(&mut self) -> Result<Stmt, CompilerError> {
        // Get struct name
        let name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
            name
        } else {
            Err(self.error("Expected struct name"))?
        };

        // Expect =
        if let Some(Token { kind: TokenKind::Equal, .. }) = self.tokens.next() {
        } else {
            Err(self.error("Expected = after struct name"))?
        }

        // Expect {
        if let Some(Token { kind: TokenKind::OpenBrace, .. }) = self.tokens.next() {
        } else {
            Err(self.error("Expected {{{{ after ="))?; // Double {{ to escape
        }

        let mut fields = Vec::new();
        
        // Parse fields until we hit }
        while let Some(token) = self.tokens.peek() {
            if token.kind == TokenKind::CloseBrace {  // Compare TokenKind instead of dereferencing
                self.tokens.next();
                break;
            }

            // Parse field name
            let field_name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
                name
            } else {
                Err(self.error("Expected field name"))?
            };

            // Expect :
            if let Some(Token { kind: TokenKind::Colon, .. }) = self.tokens.next() {
            } else {
                Err(self.error("Expected : after field name"))?
            }

            // Parse type
            let field_type = self.parse_type()?;

            fields.push((field_name, field_type));

            // Handle optional comma
            if let Some(Token { kind: TokenKind::Comma, .. }) = self.tokens.peek() {
                self.tokens.next();
            }
        }

        Ok(Stmt::StructDef {
            name,
            fields,
        })
    }

    fn parse_type(&mut self) -> Result<AstType, CompilerError> {
        match self.tokens.next().map(|t| t.kind) {
            Some(TokenKind::Identifier(type_name)) => {
                match type_name.as_str() {
                    "int" => Ok(AstType::Int),
                    "bool" => Ok(AstType::Bool),
                    "string" => Ok(AstType::String),
                    "char" => Ok(AstType::Char),
                    _ => Ok(AstType::Struct(type_name.to_string())),
                }
            },
            _ => Err(self.error("Expected type")),
        }
    }

    fn is_struct_definition(&mut self) -> bool {
        let mut iter = self.tokens.clone();
        if let Some(Token { kind: TokenKind::Identifier(_), .. }) = iter.next() {
            if let Some(Token { kind: TokenKind::Equal, .. }) = iter.next() {
                if let Some(Token { kind: TokenKind::OpenBrace, .. }) = iter.next() {
                    return true;
                }
            }
        }
        false
    }

    fn expect_token(&mut self, expected: &TokenKind) -> Result<Token, CompilerError> {
        if let Some(token) = self.tokens.next() {
            if token.kind == *expected {
                Ok(token)
            } else {
                Err(self.error(&format!(
                    "Expected {:?}, found {:?}",
                    expected,
                    token.kind
                )))
            }
        } else {
            Err(self.error("Unexpected end of file"))
        }
    }

    fn error(&mut self, message: &str) -> CompilerError {
        let current_token = self.tokens.peek()
            .cloned()
            .unwrap_or_else(|| Token::new(
                TokenKind::Identifier("EOF".to_string()),
                self.current_line(),
                self.current_column(),
                "EOF".to_string()
            ));

        let line_content = self.source_code
            .lines()
            .nth(current_token.line - 1)
            .unwrap_or("")
            .to_string();

        CompilerError::new(
            message.to_string(),
            current_token.line,
            current_token.column,
            line_content,
            ErrorType::Syntax,
        )
    }

    // Add helper methods for position tracking
    fn current_line(&mut self) -> usize {
        self.tokens.peek().map(|t| t.line).unwrap_or(1)
    }

    fn current_column(&mut self) -> usize {
        self.tokens.peek().map(|t| t.column).unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::Token;
    use crate::ast::{Expr, Stmt, Program, BinOp};

    #[test]
    fn test_parse_simple_expression() {
        let tokens = vec![
            Token::new(TokenKind::IntLiteral(42), 1, 1, "42".to_string()),
            Token::new(TokenKind::Plus, 1, 2, "+".to_string()),
            Token::new(TokenKind::IntLiteral(8), 1, 3, "8".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();  // Handle Result

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::BinaryOp(
                Box::new(Expr::IntLiteral(42)),
                BinOp::Add,
                Box::new(Expr::IntLiteral(8)),
            ))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_subtraction_expression() {
        let tokens = vec![
            Token::new(TokenKind::IntLiteral(10), 1, 1, "10".to_string()),
            Token::new(TokenKind::Minus, 1, 2, "-".to_string()),
            Token::new(TokenKind::IntLiteral(5), 1, 3, "5".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();  // Handle Result

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::BinaryOp(
                Box::new(Expr::IntLiteral(10)),
                BinOp::Subtract,
                Box::new(Expr::IntLiteral(5)),
            ))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_single_literal_expression() {
        let tokens = vec![
            Token::new(TokenKind::IntLiteral(7), 1, 1, "7".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();  // Handle Result

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::IntLiteral(7))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_unexpected_token() {
        let tokens = vec![
            Token::new(TokenKind::Plus, 1, 1, "+".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let result = parser.parse();
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("Unexpected token"));
    }

    #[test]
    fn test_parse_multiple_statements() {
        let tokens = vec![
            Token::new(TokenKind::IntLiteral(1), 1, 1, "1".to_string()),
            Token::new(TokenKind::Plus, 1, 2, "+".to_string()),
            Token::new(TokenKind::IntLiteral(2), 1, 3, "2".to_string()),
            Token::new(TokenKind::IntLiteral(3), 1, 4, "3".to_string()),
            Token::new(TokenKind::Minus, 1, 5, "-".to_string()),
            Token::new(TokenKind::IntLiteral(4), 1, 6, "4".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();  // Handle Result

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::BinaryOp(
                Box::new(Expr::IntLiteral(1)),
                BinOp::Add,
                Box::new(Expr::IntLiteral(2)),
            ))),
            Stmt::ExprStmt(Box::new(Expr::BinaryOp(
                Box::new(Expr::IntLiteral(3)),
                BinOp::Subtract,
                Box::new(Expr::IntLiteral(4)),
            ))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_boolean_literal() {
        let tokens = vec![
            Token::new(TokenKind::BoolLiteral(true), 1, 1, "true".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();  // Handle Result

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::BoolLiteral(true))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_variable_assignment() {
        let tokens = vec![
            Token::new(TokenKind::Identifier("x".to_string()), 1, 1, "x".to_string()),
            Token::new(TokenKind::Equal, 1, 2, "=".to_string()),
            Token::new(TokenKind::IntLiteral(10), 1, 3, "10".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();  // Handle Result

        let expected_ast = Program::new(vec![
            Stmt::VarAssign {
                name: "x".to_string(),
                expr: Box::new(Expr::IntLiteral(10)),
            },
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_struct_definition() {
        let tokens = vec![
            Token::new(TokenKind::Identifier("Point".to_string()), 1, 1, "Point".to_string()),
            Token::new(TokenKind::Equal, 1, 2, "=".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 3, "{".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 1, 4, "x".to_string()),
            Token::new(TokenKind::Colon, 1, 5, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 1, 6, "int".to_string()),
            Token::new(TokenKind::Comma, 1, 7, ",".to_string()),
            Token::new(TokenKind::Identifier("y".to_string()), 1, 8, "y".to_string()),
            Token::new(TokenKind::Colon, 1, 9, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 1, 10, "int".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 11, "}".to_string()),
        ];

        let mut parser = Parser::new(tokens, "".to_string());
        let result = parser.parse_struct_definition();

        // Assert that the result is Ok and compare the unwrapped value
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Stmt::StructDef {
                name: "Point".to_string(),
                fields: vec![
                    ("x".to_string(), AstType::Int),
                    ("y".to_string(), AstType::Int),
                ],
            }
        );
    }

    #[test]
    fn test_parse_function_definition() {
        let tokens = vec![
            Token::new(TokenKind::Fn, 1, 1, "fn".to_string()),
            Token::new(TokenKind::Identifier("add".to_string()), 1, 2, "add".to_string()),
            Token::new(TokenKind::OpenParen, 1, 3, "(".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 1, 4, "x".to_string()),
            Token::new(TokenKind::Colon, 1, 5, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 1, 6, "int".to_string()),
            Token::new(TokenKind::Comma, 1, 7, ",".to_string()),
            Token::new(TokenKind::Identifier("y".to_string()), 1, 8, "y".to_string()),
            Token::new(TokenKind::Colon, 1, 9, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 1, 10, "int".to_string()),
            Token::new(TokenKind::CloseParen, 1, 11, ")".to_string()),
            Token::new(TokenKind::Arrow, 1, 12, "->".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 1, 13, "int".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 14, "{".to_string()),
            Token::new(TokenKind::Return, 1, 15, "return".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 1, 16, "x".to_string()),
            Token::new(TokenKind::Plus, 1, 17, "+".to_string()),
            Token::new(TokenKind::Identifier("y".to_string()), 1, 18, "y".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 19, "}".to_string()),
        ];

        let mut parser = Parser::new(tokens, "".to_string());
        let result = parser.parse_function_definition();

        // Assert that the result is Ok and compare the unwrapped value
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            Stmt::FuncDef {
                func_decl: FuncDecl {
                    name: "add".to_string(),
                    params: vec![
                        ("x".to_string(), AstType::Int),
                        ("y".to_string(), AstType::Int),
                    ],
                    return_type: AstType::Int,
                },
                body: Box::new(Stmt::Block(vec![
                    Stmt::Return(Box::new(Expr::BinaryOp(
                        Box::new(Expr::Variable("x".to_string())),
                        BinOp::Add,
                        Box::new(Expr::Variable("y".to_string())),
                    ))),
                ])),
            }
        );
    }

    #[test]
    fn test_parse_struct_with_different_types() {
        let tokens = vec![
            Token::new(TokenKind::Identifier("User".to_string()), 1, 1, "User".to_string()),
            Token::new(TokenKind::Equal, 1, 2, "=".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 3, "{".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 1, 4, "name".to_string()),
            Token::new(TokenKind::Colon, 1, 5, ":".to_string()),
            Token::new(TokenKind::Identifier("string".to_string()), 1, 6, "string".to_string()),
            Token::new(TokenKind::Comma, 1, 7, ",".to_string()),
            Token::new(TokenKind::Identifier("age".to_string()), 1, 8, "age".to_string()),
            Token::new(TokenKind::Colon, 1, 9, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 1, 10, "int".to_string()),
            Token::new(TokenKind::Comma, 1, 11, ",".to_string()),
            Token::new(TokenKind::Identifier("active".to_string()), 1, 12, "active".to_string()),
            Token::new(TokenKind::Colon, 1, 13, ":".to_string()),
            Token::new(TokenKind::Identifier("bool".to_string()), 1, 14, "bool".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 15, "}".to_string()),
        ];

        let mut parser = Parser::new(tokens, "".to_string());
        let result = parser.parse_struct_definition().unwrap(); // Unwrap the Result

        let expected = Stmt::StructDef {
            name: "User".to_string(),
            fields: vec![
                ("name".to_string(), AstType::String),
                ("age".to_string(), AstType::Int),
                ("active".to_string(), AstType::Bool),
            ],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_struct_missing_colon() {
        let tokens = vec![
            Token::new(TokenKind::Identifier("Point".to_string()), 1, 1, "Point".to_string()),
            Token::new(TokenKind::Equal, 1, 2, "=".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 3, "{".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 1, 4, "x".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 1, 5, "int".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 6, "}".to_string()),
        ];

        let mut parser = Parser::new(tokens, "".to_string());
        let result = parser.parse_struct_definition();
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("Expected : after field name"));
    }
}
