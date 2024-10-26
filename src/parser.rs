use std::iter::Peekable;
use std::vec::IntoIter;
use crate::token::Token;
use crate::ast::{Expr, Stmt, Program, BinOp, AstType, FuncDecl};

pub struct Parser {
    tokens: Peekable<IntoIter<Token>>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens: tokens.into_iter().peekable(),
        }
    }

    pub fn parse(&mut self) -> Program {
        let mut statements = Vec::new();
        while let Some(_) = self.tokens.peek() {
            statements.push(self.parse_statement());
        }
        Program::new(statements)
    }

    fn parse_statement(&mut self) -> Stmt {
        match self.tokens.peek() {
            Some(Token::Fn) => {
                return self.parse_function_definition();
            }
            Some(Token::Return) => {
                self.tokens.next(); // consume Return token
                let expr = self.parse_expression();
                return Stmt::Return(Box::new(expr));
            }
            Some(Token::Identifier(_)) => {
                if self.is_struct_definition() {
                    return self.parse_struct_definition();
                } else if self.is_assignment() {
                    return self.parse_assignment();
                }
            }
            _ => {}
        }
        let expr = self.parse_expression();
        Stmt::ExprStmt(Box::new(expr))
    }

    fn is_assignment(&mut self) -> bool {
        if let Some(Token::Identifier(_)) = self.tokens.peek() {
            let mut iter = self.tokens.clone();
            iter.next(); // Skip identifier
            if let Some(Token::Equal) = iter.peek() {
                return true;
            }
        }
        false
    }

    fn parse_assignment(&mut self) -> Stmt {
        if let Some(Token::Identifier(name)) = self.tokens.next() {
            self.tokens.next(); // Skip '='
            let expr = self.parse_expression();
            Stmt::VarAssign {
                name,
                expr: Box::new(expr),
            }
        } else {
            panic!("Expected identifier for assignment");
        }
    }

    fn parse_expression(&mut self) -> Expr {
        let left = self.parse_primary();
        if let Some(op) = self.parse_operator() {
            let right = self.parse_primary();
            Expr::BinaryOp(Box::new(left), op, Box::new(right))
        } else {
            left
        }
    }

    fn parse_primary(&mut self) -> Expr {
        match self.tokens.next() {
            Some(Token::IntLiteral(value)) => Expr::IntLiteral(value),
            Some(Token::BoolLiteral(value)) => Expr::BoolLiteral(value),
            Some(Token::Identifier(name)) => {
                // Check if this is a function call
                if let Some(&Token::OpenParen) = self.tokens.peek() {
                    self.tokens.next(); // consume (
                    let mut args = Vec::new();
                    
                    while let Some(token) = self.tokens.peek() {
                        if *token == Token::CloseParen {
                            self.tokens.next(); // consume )
                            break;
                        }
                        
                        args.push(self.parse_expression());
                        
                        if let Some(&Token::Comma) = self.tokens.peek() {
                            self.tokens.next(); // consume comma
                        }
                    }
                    
                    Expr::FuncCall(name, args)
                } else {
                    Expr::Variable(name)
                }
            }
            _ => panic!("Unexpected token: {:?}", self.tokens.peek()),
        }
    }

    fn parse_operator(&mut self) -> Option<BinOp> {
        match self.tokens.peek() {
            Some(Token::Plus) => {
                self.tokens.next();
                Some(BinOp::Add)
            }
            Some(Token::Minus) => {
                self.tokens.next();
                Some(BinOp::Subtract)
            }
            Some(Token::Star) => {
                self.tokens.next();
                Some(BinOp::Multiply)
            }
            Some(Token::Slash) => {
                self.tokens.next();
                Some(BinOp::Divide)
            }
            Some(Token::Percent) => {
                self.tokens.next();
                Some(BinOp::Modulus)
            }
            _ => None,
        }
    }

    fn parse_function_definition(&mut self) -> Stmt {
        // Assume the current token is the function keyword
        self.tokens.next(); // Consume 'fn'
        
        let name = if let Some(Token::Identifier(name)) = self.tokens.next() {
            name
        } else {
            panic!("Expected function name");
        };

        // Parse parameter list and return type
        let params = self.parse_parameters();
        let return_type = self.parse_return_type();

        // Create the function declaration
        let func_decl = FuncDecl {
            name,
            params,
            return_type,
        };

        // Parse function body
        let body = self.parse_block();

        Stmt::FuncDef {
            func_decl,
            body: Box::new(body),
        }
    }

    fn parse_parameters(&mut self) -> Vec<(String, AstType)> {
        let mut params = Vec::new();
        
        // Expect opening parenthesis
        self.expect_token(&Token::OpenParen);
        
        // Parse parameters until we hit )
        while let Some(token) = self.tokens.peek() {
            if *token == Token::CloseParen {
                self.tokens.next(); // consume the )
                break;
            }

            // Parse parameter name
            let param_name = if let Some(Token::Identifier(name)) = self.tokens.next() {
                name
            } else {
                panic!("Expected parameter name");
            };

            // Expect :
            self.expect_token(&Token::Colon);

            // Parse parameter type
            let param_type = self.parse_type();
            params.push((param_name, param_type));

            // Handle optional comma
            if let Some(Token::Comma) = self.tokens.peek() {
                self.tokens.next();
            }
        }

        params
    }

    fn parse_return_type(&mut self) -> AstType {
        // Check for ->
        if let Some(Token::Arrow) = self.tokens.peek() {
            self.tokens.next();
            self.parse_type()
        } else {
            AstType::Void
        }
    }

    fn parse_block(&mut self) -> Stmt {
        // Expect opening brace
        self.expect_token(&Token::OpenBrace);
        
        let mut statements = Vec::new();
        
        // Parse statements until we hit }
        while let Some(token) = self.tokens.peek() {
            if *token == Token::CloseBrace {
                self.tokens.next(); // consume the }
                break;
            }
            statements.push(self.parse_statement());
        }

        Stmt::Block(statements)
    }

    fn parse_struct_definition(&mut self) -> Stmt {
        // Get struct name
        let name = if let Some(Token::Identifier(name)) = self.tokens.next() {
            name
        } else {
            panic!("Expected struct name");
        };

        // Expect =
        if let Some(Token::Equal) = self.tokens.next() {
        } else {
            panic!("Expected = after struct name");
        }

        // Expect {
        if let Some(Token::OpenBrace) = self.tokens.next() {
        } else {
            panic!("Expected {{{{ after ="); // Double {{ to escape
        }

        let mut fields = Vec::new();
        
        // Parse fields until we hit }
        while let Some(token) = self.tokens.peek() {
            if *token == Token::CloseBrace {
                self.tokens.next(); // consume the }
                break;
            }

            // Parse field name
            let field_name = if let Some(Token::Identifier(name)) = self.tokens.next() {
                name
            } else {
                panic!("Expected field name");
            };

            // Expect :
            if let Some(Token::Colon) = self.tokens.next() {
            } else {
                panic!("Expected : after field name");
            }

            // Parse type
            let field_type = self.parse_type();

            fields.push((field_name, field_type));

            // Handle optional comma
            if let Some(Token::Comma) = self.tokens.peek() {
                self.tokens.next();
            }
        }

        Stmt::StructDef {
            name,
            fields,
        }
    }

    fn parse_type(&mut self) -> AstType {
        match self.tokens.next() {
            Some(Token::Identifier(type_name)) => {
                match type_name.as_str() {
                    "int" => AstType::Int,
                    "bool" => AstType::Bool,
                    "string" => AstType::String,
                    "char" => AstType::Char,
                    _ => AstType::Struct(type_name),
                }
            },
            _ => panic!("Expected type"),
        }
    }

    fn is_struct_definition(&mut self) -> bool {
        let mut iter = self.tokens.clone();
        if let Some(Token::Identifier(_)) = iter.next() {
            if let Some(Token::Equal) = iter.next() {
                if let Some(Token::OpenBrace) = iter.next() {
                    return true;
                }
            }
        }
        false
    }

    fn expect_token(&mut self, expected: &Token) {
        if let Some(token) = self.tokens.next() {
            if token != *expected {
                panic!("Expected {:?}, got {:?}", expected, token);
            }
        } else {
            panic!("Expected {:?}, got end of input", expected);
        }
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
            Token::IntLiteral(42),
            Token::Plus,
            Token::IntLiteral(8),
        ];
        let mut parser = Parser::new(tokens);
        let program = parser.parse();

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
            Token::IntLiteral(10),
            Token::Minus,
            Token::IntLiteral(5),
        ];
        let mut parser = Parser::new(tokens);
        let program = parser.parse();

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
            Token::IntLiteral(7),
        ];
        let mut parser = Parser::new(tokens);
        let program = parser.parse();

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::IntLiteral(7))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    #[should_panic(expected = "Unexpected token")]
    fn test_parse_unexpected_token() {
        let tokens = vec![
            Token::Plus,
        ];
        let mut parser = Parser::new(tokens);
        parser.parse();
    }

    #[test]
    fn test_parse_multiple_statements() {
        let tokens = vec![
            Token::IntLiteral(1),
            Token::Plus,
            Token::IntLiteral(2),
            Token::IntLiteral(3),
            Token::Minus,
            Token::IntLiteral(4),
        ];
        let mut parser = Parser::new(tokens);
        let program = parser.parse();

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
            Token::BoolLiteral(true),
        ];
        let mut parser = Parser::new(tokens);
        let program = parser.parse();

        let expected_ast = Program::new(vec![
            Stmt::ExprStmt(Box::new(Expr::BoolLiteral(true))),
        ]);

        assert_eq!(program, expected_ast);
    }

    #[test]
    fn test_parse_variable_assignment() {
        let tokens = vec![
            Token::Identifier("x".to_string()),
            Token::Equal,
            Token::IntLiteral(10),
        ];
        let mut parser = Parser::new(tokens);
        let program = parser.parse();

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
            Token::Identifier("Point".to_string()),
            Token::Equal,
            Token::OpenBrace,
            Token::Identifier("x".to_string()),
            Token::Colon,
            Token::Identifier("int".to_string()),
            Token::Comma,
            Token::Identifier("y".to_string()),
            Token::Colon,
            Token::Identifier("int".to_string()),
            Token::CloseBrace,
        ];

        let mut parser = Parser::new(tokens);
        let result = parser.parse_struct_definition();

        let expected = Stmt::StructDef {
            name: "Point".to_string(),
            fields: vec![
                ("x".to_string(), AstType::Int),
                ("y".to_string(), AstType::Int),
            ],
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_function_definition() {
        let tokens = vec![
            Token::Fn,
            Token::Identifier("add".to_string()),
            Token::OpenParen,
            Token::Identifier("x".to_string()),
            Token::Colon,
            Token::Identifier("int".to_string()),
            Token::Comma,
            Token::Identifier("y".to_string()),
            Token::Colon,
            Token::Identifier("int".to_string()),
            Token::CloseParen,
            Token::Arrow,
            Token::Identifier("int".to_string()),
            Token::OpenBrace,
            Token::Return,
            Token::Identifier("x".to_string()),
            Token::Plus,
            Token::Identifier("y".to_string()),
            Token::CloseBrace,
        ];

        let mut parser = Parser::new(tokens);
        let result = parser.parse_function_definition();

        let expected = Stmt::FuncDef {
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
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_struct_with_different_types() {
        let tokens = vec![
            Token::Identifier("User".to_string()),
            Token::Equal,
            Token::OpenBrace,
            Token::Identifier("name".to_string()),
            Token::Colon,
            Token::Identifier("string".to_string()),
            Token::Comma,
            Token::Identifier("age".to_string()),
            Token::Colon,
            Token::Identifier("int".to_string()),
            Token::Comma,
            Token::Identifier("active".to_string()),
            Token::Colon,
            Token::Identifier("bool".to_string()),
            Token::CloseBrace,
        ];

        let mut parser = Parser::new(tokens);
        let result = parser.parse_struct_definition();

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
    #[should_panic(expected = "Expected : after field name")]
    fn test_parse_struct_missing_colon() {
        let tokens = vec![
            Token::Identifier("Point".to_string()),
            Token::Equal,
            Token::OpenBrace,
            Token::Identifier("x".to_string()),
            Token::Identifier("int".to_string()),
            Token::CloseBrace,
        ];

        let mut parser = Parser::new(tokens);
        parser.parse_struct_definition();
    }
}
