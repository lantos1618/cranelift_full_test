use std::iter::Peekable;
use std::vec::IntoIter;
use crate::token::Token;
use crate::ast::{Expr, Stmt, Program, BinOp, AstType};

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
        if let Some(Token::Identifier(_)) = self.tokens.peek() {
            if self.is_assignment() {
                return self.parse_assignment();
            }
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
            Some(Token::Identifier(name)) => Expr::Variable(name),
            _ => panic!("Unexpected token"),
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

        // Parse function body
        let body = self.parse_block();

        Stmt::FuncDef {
            name,
            params,
            return_type,
            body: Box::new(body),
        }
    }

    fn parse_parameters(&mut self) -> Vec<(String, AstType)> {
        // Implement parameter parsing logic
        vec![]
    }

    fn parse_return_type(&mut self) -> AstType {
        // Implement return type parsing logic
        AstType::Void
    }

    fn parse_block(&mut self) -> Stmt {
        // Implement block parsing logic
        Stmt::Block(vec![])
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
}
