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
                self.expect_token(&TokenKind::Semicolon)?;
                Ok(Stmt::Return(Box::new(expr)))
            }
            Some(TokenKind::Let) => {
                self.parse_let_statement()
            }
            Some(TokenKind::If) => {
                self.parse_if_statement()
            }
            Some(TokenKind::Struct) => {
                self.tokens.next(); // consume struct keyword
                let name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
                    name
                } else {
                    return Err(self.error("Expected struct name"));
                };
                self.parse_struct_definition(name)
            }
            Some(TokenKind::Identifier(_)) => {
                if self.is_assignment() {
                    self.parse_assignment()
                } else {
                    self.parse_expr_statement()
                }
            }
            _ => {
                self.parse_expr_statement()
            }
        }
    }

    #[allow(dead_code)]
    fn is_assignment(&mut self) -> bool {
        let mut iter = self.tokens.clone();
        if let Some(Token { kind: TokenKind::Identifier(_), .. }) = iter.next() {
            matches!(iter.next(), Some(Token { kind: TokenKind::Equal, .. }))
        } else {
            false
        }
    }

    #[allow(dead_code)]
    fn parse_assignment(&mut self) -> Result<Stmt, CompilerError> {
        let name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
            name
        } else {
            return Err(self.error("Expected identifier for assignment"));
        };
        
        self.expect_token(&TokenKind::Equal)?;
        let expr = self.parse_expression()?;
        self.expect_token(&TokenKind::Semicolon)?;
        
        Ok(Stmt::VarAssign {
            name,
            expr: Box::new(expr),
        })
    }

    fn parse_expression(&mut self) -> Result<Expr, CompilerError> {
        let mut expr = self.parse_primary()?;
        
        while let Some(token) = self.tokens.peek() {
            match &token.kind {
                TokenKind::OpenBrace => {
                    if let Expr::Variable(struct_name) = expr.clone() {
                        self.tokens.next(); // consume {
                        expr = self.parse_struct_init(struct_name)?;
                    } else {
                        break;
                    }
                }
                TokenKind::Dot => {
                    self.tokens.next(); // consume .
                    let field_name = match self.tokens.next() {
                        Some(Token { kind: TokenKind::Identifier(name), .. }) => name,
                        _ => return Err(self.error("Expected field name after dot")),
                    };
                    expr = Expr::StructAccess(Box::new(expr), field_name);
                }
                TokenKind::Plus | TokenKind::Minus | TokenKind::Star | TokenKind::Slash | TokenKind::Percent | TokenKind::LessThan | TokenKind::LessThanEqual | TokenKind::GreaterThan | TokenKind::GreaterThanEqual => {
                    if let Some(op) = self.parse_operator() {
                        let right = self.parse_primary()?;
                        expr = Expr::BinaryOp(Box::new(expr), op, Box::new(right));
                    }
                }
                _ => break,
            }
        }
        
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, CompilerError> {
        let mut expr = match self.tokens.next().map(|t| t.kind) {
            Some(TokenKind::Identifier(name)) => {
                // Check if this is a function call or struct initialization
                if let Some(token) = self.tokens.peek() {
                    match token.kind {
                        TokenKind::OpenParen => {
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
                            
                            Expr::FuncCall(name.to_string(), args)
                        }
                        TokenKind::OpenBrace => {
                            self.tokens.next(); // consume {
                            self.parse_struct_init(name)?
                        }
                        _ => Expr::Variable(name),
                    }
                } else {
                    Expr::Variable(name)
                }
            }
            Some(TokenKind::IntLiteral(value)) => Expr::IntLiteral(value.into()),
            Some(TokenKind::BoolLiteral(value)) => Expr::BoolLiteral(value),
            _ => return Err(self.error("Unexpected token")),
        };

        // Handle field access with dot operator
        while let Some(Token { kind: TokenKind::Dot, .. }) = self.tokens.peek() {
            self.tokens.next(); // consume .
            
            let field_name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
                name
            } else {
                return Err(self.error("Expected field name after dot"));
            };
            
            expr = Expr::StructAccess(Box::new(expr), field_name);
        }

        Ok(expr)
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
            Some(TokenKind::LessThan) => {
                self.tokens.next();
                Some(BinOp::LessThan)
            }
            Some(TokenKind::LessThanEqual) => {
                self.tokens.next();
                Some(BinOp::LessThanEqual)
            }
            Some(TokenKind::GreaterThan) => {
                self.tokens.next();
                Some(BinOp::GreaterThan)
            }
            Some(TokenKind::GreaterThanEqual) => {
                self.tokens.next();
                Some(BinOp::GreaterThanEqual)
            }
            _ => None,
        }
    }

    fn parse_function_definition(&mut self) -> Result<Stmt, CompilerError> {
        // Consume 'fn' keyword
        self.expect_token(&TokenKind::Fn)?;
        
        // Parse function name
        let name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
            name
        } else {
            return Err(self.error("Expected function name"));
        };

        // Parse parameter list
        self.expect_token(&TokenKind::OpenParen)?;
        let mut params = Vec::new();
        
        while let Some(token) = self.tokens.peek() {
            if token.kind == TokenKind::CloseParen {
                self.tokens.next(); // consume )
                break;
            }

            // Parse parameter name
            let param_name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
                name
            } else {
                return Err(self.error("Expected parameter name"));
            };

            // Expect :
            self.expect_token(&TokenKind::Colon)?;

            // Parse parameter type
            let param_type = self.parse_type()?;
            params.push((param_name, param_type));

            // Handle comma
            if let Some(token) = self.tokens.peek() {
                if token.kind == TokenKind::Comma {
                    self.tokens.next();
                } else if token.kind != TokenKind::CloseParen {
                    return Err(self.error("Expected comma or closing parenthesis"));
                }
            }
        }

        // Parse return type
        let return_type = if let Some(token) = self.tokens.peek() {
            if token.kind == TokenKind::Arrow {
                self.tokens.next(); // consume ->
                self.parse_type()?
            } else {
                AstType::Void
            }
        } else {
            return Err(self.error("Unexpected end of input"));
        };

        // Create function declaration
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

    fn parse_block(&mut self) -> Result<Stmt, CompilerError> {
        self.expect_token(&TokenKind::OpenBrace)?;
        
        let mut statements = Vec::new();
        
        while let Some(token) = self.tokens.peek() {
            if token.kind == TokenKind::CloseBrace {
                self.tokens.next(); // consume }
                return Ok(Stmt::Block(statements));
            }
            statements.push(self.parse_statement()?);
        }
        
        Err(self.error("Unclosed block"))
    }

    fn parse_struct_definition(&mut self, name: String) -> Result<Stmt, CompilerError> {
        self.expect_token(&TokenKind::OpenBrace)?;
        
        let mut fields = Vec::new();
        
        loop {
            match self.tokens.peek() {
                Some(token) if token.kind == TokenKind::CloseBrace => {
                    self.tokens.next(); // consume }
                    self.expect_token(&TokenKind::Semicolon)?;
                    return Ok(Stmt::StructDef {
                        name,
                        fields,
                    });
                }
                Some(Token { kind: TokenKind::Identifier(_), .. }) => {
                    let field_name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
                        name
                    } else {
                        return Err(self.error("Expected field name"));
                    };
                    
                    // Check for colon with specific error message
                    if let Err(_) = self.expect_token(&TokenKind::Colon) {
                        return Err(self.error("Expected : after field name"));
                    }
                    
                    let field_type = self.parse_type()?;
                    fields.push((field_name, field_type));
                    
                    // Handle optional comma
                    if let Some(token) = self.tokens.peek() {
                        if token.kind == TokenKind::Comma {
                            self.tokens.next();
                        }
                    }
                }
                Some(_) => return Err(self.error("Expected field name")),
                None => return Err(self.error("Unclosed struct definition")),
            }
        }
    }

    fn parse_struct_init(&mut self, struct_name: String) -> Result<Expr, CompilerError> {
        let mut fields = Vec::new();
        
        loop {
            match self.tokens.peek() {
                Some(token) if token.kind == TokenKind::CloseBrace => {
                    self.tokens.next(); // consume }
                    return Ok(Expr::StructInit {
                        struct_name,
                        fields,
                    });
                }
                Some(Token { kind: TokenKind::Identifier(_), .. }) => {
                    let field_name = if let Some(Token { kind: TokenKind::Identifier(name), .. }) = self.tokens.next() {
                        name
                    } else {
                        return Err(self.error("Expected field name"));
                    };
                    
                    if let Err(_) = self.expect_token(&TokenKind::Colon) {
                        return Err(self.error("Expected : after field name"));
                    }
                    
                    let field_value = self.parse_expression()?;
                    fields.push((field_name, field_value));
                    
                    match self.tokens.peek() {
                        Some(token) if token.kind == TokenKind::Comma => {
                            self.tokens.next(); // consume comma
                        }
                        Some(token) if token.kind == TokenKind::CloseBrace => continue,
                        _ => return Err(self.error("Expected comma or closing brace")),
                    }
                }
                Some(_) => return Err(self.error("Expected field name")),
                None => return Err(self.error("Unexpected end of input")),
            }
        }
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
        iter.next(); // Skip identifier
        matches!(iter.next(), Some(Token { kind: TokenKind::OpenBrace, .. }))
    }

    fn expect_token(&mut self, expected: &TokenKind) -> Result<Token, CompilerError> {
        if let Some(token) = self.tokens.next() {
            if token.kind == *expected {
                Ok(token)
            } else {
                match expected {
                    TokenKind::OpenParen => {
                        Err(self.error("Expected ("))
                    }
                    _ => Err(self.error(&format!(
                        "Expected {:?}, found {:?}",
                        expected,
                        token.kind
                    )))
                }
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

    fn parse_expr_statement(&mut self) -> Result<Stmt, CompilerError> {
        let expr = self.parse_expression()?;
        self.expect_token(&TokenKind::Semicolon)?;
        Ok(Stmt::ExprStmt(Box::new(expr)))
    }

    fn parse_let_statement(&mut self) -> Result<Stmt, CompilerError> {
        self.tokens.next(); // consume Let token
        let name = match self.tokens.next() {
            Some(Token { kind: TokenKind::Identifier(name), .. }) => name,
            _ => return Err(self.error("Expected identifier after let")),
        };
        self.expect_token(&TokenKind::Colon)?;
        let var_type = self.parse_type()?;
        self.expect_token(&TokenKind::Equal)?;
        let expr = self.parse_expression()?;
        self.expect_token(&TokenKind::Semicolon)?;
        Ok(Stmt::VarDecl {
            name,
            var_type,
            init_expr: Some(Box::new(expr)),
        })
    }

    fn parse_if_statement(&mut self) -> Result<Stmt, CompilerError> {
        // Consume 'if' token
        self.expect_token(&TokenKind::If)?;
        
        // Expect opening parenthesis
        self.expect_token(&TokenKind::OpenParen)?;
        
        // Parse condition
        let condition = self.parse_expression()?;
        
        // Expect closing parenthesis
        self.expect_token(&TokenKind::CloseParen)?;
        
        // Parse then branch
        let then_branch = Box::new(self.parse_block()?);
        
        // Check for else
        let else_branch = if let Some(TokenKind::Else) = self.tokens.peek().map(|t| &t.kind) {
            self.tokens.next(); // consume 'else'
            Some(Box::new(self.parse_block()?))
        } else {
            None
        };

        Ok(Stmt::If {
            condition: Box::new(condition),
            then_branch,
            else_branch,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::{Token, TokenKind};
    use crate::ast::{Expr, Stmt, Program, BinOp};

    #[test]
    fn test_parse_simple_expression() {
        let tokens = vec![
            Token::new(TokenKind::IntLiteral(42), 1, 1, "42".to_string()),
            Token::new(TokenKind::Plus, 1, 2, "+".to_string()),
            Token::new(TokenKind::IntLiteral(8), 1, 3, "8".to_string()),
            Token::new(TokenKind::Semicolon, 1, 4, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();

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
            Token::new(TokenKind::Semicolon, 1, 4, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();

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
            Token::new(TokenKind::Semicolon, 1, 2, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::IntLiteral(7))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_boolean_literal() {
        let tokens = vec![
            Token::new(TokenKind::BoolLiteral(true), 1, 1, "true".to_string()),
            Token::new(TokenKind::Semicolon, 1, 2, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::BoolLiteral(true))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_multiple_statements() {
        let tokens = vec![
            Token::new(TokenKind::IntLiteral(1), 1, 1, "1".to_string()),
            Token::new(TokenKind::Plus, 1, 2, "+".to_string()),
            Token::new(TokenKind::IntLiteral(2), 1, 3, "2".to_string()),
            Token::new(TokenKind::Semicolon, 1, 4, ";".to_string()),
            Token::new(TokenKind::IntLiteral(3), 1, 5, "3".to_string()),
            Token::new(TokenKind::Minus, 1, 6, "-".to_string()),
            Token::new(TokenKind::IntLiteral(4), 1, 7, "4".to_string()),
            Token::new(TokenKind::Semicolon, 1, 8, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();

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
    fn test_parse_variable_assignment() {
        let tokens = vec![
            Token::new(TokenKind::Identifier("x".to_string()), 1, 1, "x".to_string()),
            Token::new(TokenKind::Equal, 1, 3, "=".to_string()),
            Token::new(TokenKind::IntLiteral(10), 1, 5, "10".to_string()),
            Token::new(TokenKind::Semicolon, 1, 7, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let result = parser.parse_statement().unwrap();

        assert_eq!(
            result,
            Stmt::VarAssign {
                name: "x".to_string(),
                expr: Box::new(Expr::IntLiteral(10)),
            }
        );
    }

    #[test]
    fn test_parse_struct_definition() {
        let tokens = vec![
            Token::new(TokenKind::Struct, 1, 1, "struct".to_string()),
            Token::new(TokenKind::Identifier("Point".to_string()), 1, 7, "Point".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 13, "{".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 2, 5, "x".to_string()),
            Token::new(TokenKind::Colon, 2, 6, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 2, 8, "int".to_string()),
            Token::new(TokenKind::Comma, 2, 11, ",".to_string()),
            Token::new(TokenKind::Identifier("y".to_string()), 3, 5, "y".to_string()),
            Token::new(TokenKind::Colon, 3, 6, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 3, 8, "int".to_string()),
            Token::new(TokenKind::CloseBrace, 4, 1, "}".to_string()),
            Token::new(TokenKind::Semicolon, 4, 2, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "test source".to_string());
        let stmt = parser.parse_statement().unwrap();
        
        match stmt {
            Stmt::StructDef { name, fields } => {
                assert_eq!(name, "Point");
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "x");
                assert_eq!(fields[1].0, "y");
            }
            _ => panic!("Expected struct definition"),
        }
    }

    #[test]
    fn test_parse_struct_with_different_types() {
        let tokens = vec![
            Token::new(TokenKind::Struct, 1, 1, "struct".to_string()),
            Token::new(TokenKind::Identifier("User".to_string()), 1, 7, "User".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 13, "{".to_string()),
            Token::new(TokenKind::Identifier("name".to_string()), 2, 5, "name".to_string()),
            Token::new(TokenKind::Colon, 2, 6, ":".to_string()),
            Token::new(TokenKind::Identifier("string".to_string()), 2, 8, "string".to_string()),
            Token::new(TokenKind::Comma, 2, 14, ",".to_string()),
            Token::new(TokenKind::Identifier("age".to_string()), 3, 5, "age".to_string()),
            Token::new(TokenKind::Colon, 3, 6, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 3, 8, "int".to_string()),
            Token::new(TokenKind::CloseBrace, 4, 1, "}".to_string()),
            Token::new(TokenKind::Semicolon, 4, 2, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let stmt = parser.parse_statement().unwrap();
        
        match stmt {
            Stmt::StructDef { name, fields } => {
                assert_eq!(name, "User");
                assert_eq!(fields.len(), 2);
            }
            _ => panic!("Expected struct definition"),
        }
    }

    #[test]
    fn test_parse_struct_missing_colon() {
        let tokens = vec![
            Token::new(TokenKind::Struct, 1, 1, "struct".to_string()),
            Token::new(TokenKind::Identifier("Point".to_string()), 1, 7, "Point".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 13, "{".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 1, 14, "x".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 1, 16, "int".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 20, "}".to_string()),
            Token::new(TokenKind::Semicolon, 1, 21, ";".to_string()),
        ];

        let mut parser = Parser::new(tokens, "struct Point { x int }".to_string());
        let result = parser.parse_statement();
        
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.message, "Expected : after field name");
    }

    #[test]
    fn test_parse_struct_field_access() {
        let tokens = vec![
            Token::new(TokenKind::Identifier("point".to_string()), 1, 1, "point".to_string()),
            Token::new(TokenKind::Dot, 1, 6, ".".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 1, 7, "x".to_string()),
            Token::new(TokenKind::Semicolon, 1, 8, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::StructAccess(
                Box::new(Expr::Variable("point".to_string())),
                "x".to_string(),
            ))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_struct_initialization() {
        let tokens = vec![
            Token::new(TokenKind::Let, 1, 1, "let".to_string()),
            Token::new(TokenKind::Identifier("p".to_string()), 1, 5, "p".to_string()),
            Token::new(TokenKind::Colon, 1, 6, ":".to_string()),
            Token::new(TokenKind::Identifier("Point".to_string()), 1, 8, "Point".to_string()),
            Token::new(TokenKind::Equal, 1, 14, "=".to_string()),
            Token::new(TokenKind::Identifier("Point".to_string()), 1, 16, "Point".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 21, "{".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 1, 22, "x".to_string()),
            Token::new(TokenKind::Colon, 1, 23, ":".to_string()),
            Token::new(TokenKind::IntLiteral(3), 1, 25, "3".to_string()),
            Token::new(TokenKind::Comma, 1, 26, ",".to_string()),
            Token::new(TokenKind::Identifier("y".to_string()), 1, 28, "y".to_string()),
            Token::new(TokenKind::Colon, 1, 29, ":".to_string()),
            Token::new(TokenKind::IntLiteral(4), 1, 31, "4".to_string()),
            Token::new(TokenKind::CloseBrace, 1, 32, "}".to_string()),
            Token::new(TokenKind::Semicolon, 1, 33, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let stmt = parser.parse_statement().unwrap();

        match stmt {
            Stmt::VarDecl { name, var_type, init_expr } => {
                assert_eq!(name, "p");
                assert_eq!(var_type, AstType::Struct("Point".to_string()));
                match *init_expr.unwrap() {
                    Expr::StructInit { struct_name, fields } => {
                        assert_eq!(struct_name, "Point");
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].0, "x");
                        assert_eq!(fields[1].0, "y");
                    }
                    _ => panic!("Expected struct initialization"),
                }
            }
            _ => panic!("Expected variable declaration"),
        }
    }


    #[test]
    fn test_parse_chained_field_access() {
        let tokens = vec![
            Token::new(TokenKind::Identifier("obj".to_string()), 1, 1, "obj".to_string()),
            Token::new(TokenKind::Dot, 1, 4, ".".to_string()),
            Token::new(TokenKind::Identifier("field1".to_string()), 1, 5, "field1".to_string()),
            Token::new(TokenKind::Dot, 1, 11, ".".to_string()),
            Token::new(TokenKind::Identifier("field2".to_string()), 1, 12, "field2".to_string()),
            Token::new(TokenKind::Semicolon, 1, 18, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::StructAccess(
                Box::new(Expr::StructAccess(
                    Box::new(Expr::Variable("obj".to_string())),
                    "field1".to_string(),
                )),
                "field2".to_string(),
            ))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_field_access_with_operation() {
        let tokens = vec![
            Token::new(TokenKind::Identifier("point".to_string()), 1, 1, "point".to_string()),
            Token::new(TokenKind::Dot, 1, 6, ".".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 1, 7, "x".to_string()),
            Token::new(TokenKind::Star, 1, 9, "*".to_string()),
            Token::new(TokenKind::Identifier("point".to_string()), 1, 11, "point".to_string()),
            Token::new(TokenKind::Dot, 1, 16, ".".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 1, 17, "x".to_string()),
            Token::new(TokenKind::Semicolon, 1, 18, ";".to_string()),
        ];
        let mut parser = Parser::new(tokens, "".to_string());
        let program = parser.parse().unwrap();

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::BinaryOp(
                Box::new(Expr::StructAccess(
                    Box::new(Expr::Variable("point".to_string())),
                    "x".to_string(),
                )),
                BinOp::Multiply,
                Box::new(Expr::StructAccess(
                    Box::new(Expr::Variable("point".to_string())),
                    "x".to_string(),
                )),
            ))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_struct_definition_and_expression() {
        let tokens = vec![
            // Struct definition
            Token::new(TokenKind::Struct, 1, 1, "struct".to_string()),
            Token::new(TokenKind::Identifier("Point".to_string()), 1, 7, "Point".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 13, "{".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 2, 5, "x".to_string()),
            Token::new(TokenKind::Colon, 2, 6, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 2, 8, "int".to_string()),
            Token::new(TokenKind::Comma, 2, 11, ",".to_string()),
            Token::new(TokenKind::Identifier("y".to_string()), 3, 5, "y".to_string()),
            Token::new(TokenKind::Colon, 3, 6, ":".to_string()),
            Token::new(TokenKind::Identifier("int".to_string()), 3, 8, "int".to_string()),
            Token::new(TokenKind::CloseBrace, 4, 1, "}".to_string()),
            Token::new(TokenKind::Semicolon, 4, 2, ";".to_string()),
            // Variable declaration with struct initialization
            Token::new(TokenKind::Let, 5, 1, "let".to_string()),
            Token::new(TokenKind::Identifier("p".to_string()), 5, 5, "p".to_string()),
            Token::new(TokenKind::Colon, 5, 6, ":".to_string()),
            Token::new(TokenKind::Identifier("Point".to_string()), 5, 8, "Point".to_string()),
            Token::new(TokenKind::Equal, 5, 14, "=".to_string()),
            Token::new(TokenKind::Identifier("Point".to_string()), 5, 16, "Point".to_string()),
            Token::new(TokenKind::OpenBrace, 5, 21, "{".to_string()),
            Token::new(TokenKind::Identifier("x".to_string()), 5, 22, "x".to_string()),
            Token::new(TokenKind::Colon, 5, 23, ":".to_string()),
            Token::new(TokenKind::IntLiteral(1), 5, 25, "1".to_string()),
            Token::new(TokenKind::Comma, 5, 26, ",".to_string()),
            Token::new(TokenKind::Identifier("y".to_string()), 5, 28, "y".to_string()),
            Token::new(TokenKind::Colon, 5, 29, ":".to_string()),
            Token::new(TokenKind::IntLiteral(2), 5, 31, "2".to_string()),
            Token::new(TokenKind::CloseBrace, 5, 32, "}".to_string()),
            Token::new(TokenKind::Semicolon, 5, 33, ";".to_string()),
        ];

        let mut parser = Parser::new(tokens, "test source".to_string());
        let program = parser.parse().unwrap();
        
        assert_eq!(program.statements.len(), 2);
        
        match &program.statements[0] {
            Stmt::StructDef { name, fields } => {
                assert_eq!(name, "Point");
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "x");
                assert_eq!(fields[1].0, "y");
            }
            _ => panic!("Expected struct definition"),
        }
        
        match &program.statements[1] {
            Stmt::VarDecl { name, var_type, init_expr } => {
                assert_eq!(name, "p");
                assert_eq!(var_type, &AstType::Struct("Point".to_string()));
                match init_expr.as_ref().unwrap().as_ref() {
                    Expr::StructInit { struct_name, fields } => {
                        assert_eq!(struct_name, "Point");
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].0, "x");
                        assert_eq!(fields[1].0, "y");
                    }
                    _ => panic!("Expected struct initialization"),
                }
            }
            _ => panic!("Expected variable declaration"),
        }
    }

    #[test]
    fn test_parse_if_with_parentheses() {
        let tokens = vec![
            Token::new(TokenKind::If, 1, 1, "if".to_string()),
            Token::new(TokenKind::OpenParen, 1, 4, "(".to_string()),
            Token::new(TokenKind::IntLiteral(10), 1, 5, "10".to_string()),
            Token::new(TokenKind::LessThan, 1, 8, "<".to_string()),
            Token::new(TokenKind::IntLiteral(20), 1, 10, "20".to_string()),
            Token::new(TokenKind::CloseParen, 1, 12, ")".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 14, "{".to_string()),
            Token::new(TokenKind::Return, 2, 5, "return".to_string()),
            Token::new(TokenKind::IntLiteral(1), 2, 12, "1".to_string()),
            Token::new(TokenKind::Semicolon, 2, 13, ";".to_string()),
            Token::new(TokenKind::CloseBrace, 3, 1, "}".to_string()),
        ];

        let mut parser = Parser::new(tokens, "test source".to_string());
        let stmt = parser.parse_statement().unwrap();

        let expected_ast = Stmt::If {
            condition: Box::new(Expr::BinaryOp(
                Box::new(Expr::IntLiteral(10)), BinOp::LessThan, Box::new(Expr::IntLiteral(20))
            )),
            then_branch: Box::new(Stmt::Block(vec![Stmt::Return(Box::new(Expr::IntLiteral(1)))])),
            else_branch: None,
        };
        assert_eq!(stmt, expected_ast); 
        
    }

    #[test]
    fn test_parse_if_without_parentheses() {
        let tokens = vec![
            Token::new(TokenKind::If, 1, 1, "if".to_string()),
            Token::new(TokenKind::IntLiteral(10), 1, 4, "10".to_string()),
            Token::new(TokenKind::LessThan, 1, 7, "<".to_string()),
            Token::new(TokenKind::IntLiteral(20), 1, 9, "20".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 12, "{".to_string()),
            Token::new(TokenKind::Return, 2, 5, "return".to_string()),
            Token::new(TokenKind::IntLiteral(1), 2, 12, "1".to_string()),
            Token::new(TokenKind::Semicolon, 2, 13, ";".to_string()),
            Token::new(TokenKind::CloseBrace, 3, 1, "}".to_string()),
        ];

        let mut parser = Parser::new(tokens, "test source".to_string());
        let result = parser.parse_statement();
        
        // Should fail because parentheses are required
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.message.contains("Expected ("));
    }

    #[test]
    fn test_parse_if_with_less_equal() {
        let tokens = vec![
            Token::new(TokenKind::If, 1, 1, "if".to_string()),
            Token::new(TokenKind::OpenParen, 1, 4, "(".to_string()),
            Token::new(TokenKind::Identifier("n".to_string()), 1, 5, "n".to_string()),
            Token::new(TokenKind::LessThanEqual, 1, 7, "<=".to_string()),
            Token::new(TokenKind::IntLiteral(1), 1, 10, "1".to_string()),
            Token::new(TokenKind::CloseParen, 1, 11, ")".to_string()),
            Token::new(TokenKind::OpenBrace, 1, 13, "{".to_string()),
            Token::new(TokenKind::Return, 2, 5, "return".to_string()),
            Token::new(TokenKind::Identifier("n".to_string()), 2, 12, "n".to_string()),
            Token::new(TokenKind::Semicolon, 2, 13, ";".to_string()),
            Token::new(TokenKind::CloseBrace, 3, 1, "}".to_string()),
        ];

        let mut parser = Parser::new(tokens, "test source".to_string());
        let stmt = parser.parse_statement().unwrap();

        let expected_ast = Stmt::If {
            condition: Box::new(Expr::BinaryOp(
                Box::new(Expr::Variable("n".to_string())),
                BinOp::LessThanEqual,
                Box::new(Expr::IntLiteral(1))
            )),
            then_branch: Box::new(Stmt::Block(vec![
                Stmt::Return(Box::new(Expr::Variable("n".to_string())))
            ])),
            else_branch: None,
        };
        assert_eq!(stmt, expected_ast);
    }
}
