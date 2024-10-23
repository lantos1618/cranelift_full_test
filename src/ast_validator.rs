use crate::ast::*;
use anyhow::Result;
use std::collections::HashMap;


pub struct AstValidator {
    symbol_table: HashMap<String, AstType>,
}

impl AstValidator {
    pub fn new() -> Self {
        Self {
            symbol_table: HashMap::new(),
        }
    }

    pub fn validate_program(&mut self, program: &Program) -> Result<()> {
        for stmt in &program.statements {
            self.validate_stmt(stmt)?;
        }
        Ok(())
    }

    fn validate_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::VarDecl { name, var_type, init_expr } => {
                if let Some(expr) = init_expr {
                    let expr_type = self.validate_expr(expr)?;
                    if expr_type != *var_type {
                        return Err(anyhow::anyhow!(
                            "Type mismatch in variable declaration for '{}': \
                            expected {:?}, but got {:?}",
                            name, var_type, expr_type
                        ));
                    }
                }
                self.symbol_table.insert(name.clone(), var_type.clone());
            }
            Stmt::VarAssign { name, expr } => {
                let expr_type = self.validate_expr(expr)?;
                if let Some(var_type) = self.symbol_table.get(name) {
                    if *var_type != expr_type {
                        return Err(anyhow::anyhow!(
                            "Type mismatch in assignment to '{}': \
                            variable is of type {:?}, but trying to assign {:?}",
                            name, var_type, expr_type
                        ));
                    }
                } else {
                    return Err(anyhow::anyhow!(
                        "Undefined variable '{}'. Did you forget to declare it?",
                        name
                    ));
                }
            }
            Stmt::ExprStmt(expr) => {
                self.validate_expr(expr)?;
            }
            Stmt::Return(expr) => {
                self.validate_expr(expr)?;
            }
            Stmt::Block(stmts) => {
                for stmt in stmts {
                    self.validate_stmt(stmt)?;
                }
            }
            Stmt::FuncDef { func_decl, body } => {
                // Validate function signature and body
                self.validate_func_def(func_decl, body)?;
            }
            Stmt::If { condition, then_branch, else_branch } => {
                let cond_type = self.validate_expr(condition)?;
                if cond_type != AstType::Bool {
                    return Err(anyhow::anyhow!("Condition must be a boolean"));
                }
                self.validate_stmt(then_branch)?;
                if let Some(else_stmt) = else_branch {
                    self.validate_stmt(else_stmt)?;
                }
            }
            Stmt::While { condition, body } => {
                let cond_type = self.validate_expr(condition)?;
                if cond_type != AstType::Bool {
                    return Err(anyhow::anyhow!("Condition must be a boolean"));
                }
                self.validate_stmt(body)?;
            }
            _ => unimplemented!("Statement {:?} not implemented", stmt),
        }
        Ok(())
    }

    fn validate_expr(&mut self, expr: &Expr) -> Result<AstType> {
        match expr {
            Expr::IntLiteral(_) => Ok(AstType::Int),
            Expr::BoolLiteral(_) => Ok(AstType::Bool),
            Expr::Variable(name) => {
                if let Some(var_type) = self.symbol_table.get(name) {
                    Ok(var_type.clone())
                } else {
                    Err(anyhow::anyhow!("Undefined variable `{}`", name))
                }
            }
            Expr::BinaryOp(lhs, op, rhs) => {
                let lhs_type = self.validate_expr(lhs)?.clone();
                let rhs_type = self.validate_expr(rhs)?.clone();
                if lhs_type != rhs_type {
                    return Err(anyhow::anyhow!(
                        "Type mismatch in binary operation: left side is {:?}, right side is {:?}. \
                        Both sides must be of the same type.",
                        lhs_type, rhs_type
                    ));
                }
                match op {
                    BinOp::Add | BinOp::Subtract | BinOp::Multiply | BinOp::Divide => {
                        if lhs_type == AstType::Int {
                            Ok(AstType::Int)
                        } else {
                            Err(anyhow::anyhow!(
                                "Invalid types for arithmetic operation: found {:?}, expected Int. \
                                Arithmetic operations can only be performed on integers.",
                                lhs_type
                            ))
                        }
                    }
                    BinOp::Equal | BinOp::NotEqual | BinOp::LessThan | BinOp::GreaterThan => {
                        if lhs_type == AstType::Int || lhs_type == AstType::Bool {
                            Ok(AstType::Bool)
                        } else {
                            Err(anyhow::anyhow!(
                                "Invalid types for comparison: found {:?}. \
                                Comparisons can only be performed on integers or booleans.",
                                lhs_type
                            ))
                        }
                    }
                    _ => unimplemented!("Operator {:?} not implemented", op),
                }
            }
            Expr::UnaryOp(op, expr) => {
                let expr_type = self.validate_expr(expr)?.clone();
                match op {
                    UnaryOp::Negate => {
                        if expr_type == AstType::Int {
                            Ok(AstType::Int)
                        } else {
                            Err(anyhow::anyhow!("Invalid type for negation"))
                        }
                    }
                    UnaryOp::Not => {
                        if expr_type == AstType::Bool {
                            Ok(AstType::Bool)
                        } else {
                            Err(anyhow::anyhow!("Invalid type for logical not"))
                        }
                    }
                    _ => unimplemented!("Unary operator {:?} not implemented", op),
                }
            }
            Expr::FuncCall(name, args) => {
                // First, get and check the function type
                let func_type = if let Some(ft) = self.symbol_table.get(name) {
                    ft.clone()
                } else {
                    return Err(anyhow::anyhow!(
                        "Undefined function '{}'. Available functions are: {:?}",
                        name,
                        self.symbol_table.keys().collect::<Vec<_>>()
                    ));
                };

                match func_type {
                    AstType::Function(param_types, return_type) => {
                        if args.len() != param_types.len() {
                            return Err(anyhow::anyhow!(
                                "Wrong number of arguments for function '{}': expected {}, but got {}",
                                name,
                                param_types.len(),
                                args.len()
                            ));
                        }

                        // Validate all arguments first and collect their types
                        let mut arg_types = Vec::with_capacity(args.len());
                        for arg in args {
                            let arg_type = self.validate_expr(arg)?;
                            arg_types.push(arg_type);
                        }

                        // Then check if the types match
                        for (i, (arg_type, param_type)) in arg_types.iter().zip(param_types.iter()).enumerate() {
                            if arg_type != param_type {
                                return Err(anyhow::anyhow!(
                                    "Type mismatch in argument {} of function '{}': expected {:?}, but got {:?}",
                                    i + 1,
                                    name,
                                    param_type,
                                    arg_type
                                ));
                            }
                        }

                        Ok(*return_type)
                    }
                    _ => Err(anyhow::anyhow!(
                        "'{}' is not a function. Found type {:?} instead",
                        name,
                        func_type
                    ))
                }
            }
            _ => unimplemented!("Expression {:?} not implemented", expr),
        }
    }

    fn validate_func_def(&mut self, func_decl: &FuncDecl, body: &Stmt) -> Result<()> {
        // Add function parameters to the symbol table
        for (name, param_type) in &func_decl.params {
            self.symbol_table.insert(name.clone(), param_type.clone());
        }

        // Validate the function body
        self.validate_stmt(body)?;

        // Remove function parameters from the symbol table
        for (name, _) in &func_decl.params {
            self.symbol_table.remove(name);
        }

        Ok(())
    }

    fn validate_func_call(&mut self, _name: &str, _args: &[Expr]) -> Result<AstType> {
        // Implementation here
        Ok(AstType::Int) // Or whatever appropriate return type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_program(statements: Vec<Stmt>) -> Program {
        Program { statements }
    }

    #[test]
    fn test_valid_var_declarations() {
        let mut validator = AstValidator::new();
        let program = create_test_program(vec![
            Stmt::VarDecl {
                name: "x".to_string(),
                var_type: AstType::Int,
                init_expr: Some(Box::new(Expr::IntLiteral(42))),
            },
            Stmt::VarDecl {
                name: "b".to_string(),
                var_type: AstType::Bool,
                init_expr: Some(Box::new(Expr::BoolLiteral(true))),
            },
        ]);

        assert!(validator.validate_program(&program).is_ok());
    }

    #[test]
    fn test_invalid_var_declaration() {
        let mut validator = AstValidator::new();
        let program = create_test_program(vec![
            Stmt::VarDecl {
                name: "x".to_string(),
                var_type: AstType::Int,
                init_expr: Some(Box::new(Expr::BoolLiteral(true))), // Type mismatch
            },
        ]);

        assert!(validator.validate_program(&program).is_err());
    }

    #[test]
    fn test_valid_binary_operations() {
        let mut validator = AstValidator::new();
        let program = create_test_program(vec![
            Stmt::VarDecl {
                name: "x".to_string(),
                var_type: AstType::Int,
                init_expr: Some(Box::new(Expr::BinaryOp(
                    Box::new(Expr::IntLiteral(5)),
                    BinOp::Add,
                    Box::new(Expr::IntLiteral(3)),
                ))),
            },
        ]);

        assert!(validator.validate_program(&program).is_ok());
    }

    #[test]
    fn test_invalid_binary_operation() {
        let mut validator = AstValidator::new();
        let program = create_test_program(vec![
            Stmt::ExprStmt(Box::new(Expr::BinaryOp(
                Box::new(Expr::IntLiteral(5)),
                BinOp::Add,
                Box::new(Expr::BoolLiteral(true)), // Type mismatch
            ))),
        ]);

        assert!(validator.validate_program(&program).is_err());
    }

    #[test]
    fn test_valid_if_statement() {
        let mut validator = AstValidator::new();
        let program = create_test_program(vec![
            Stmt::If {
                condition: Box::new(Expr::BoolLiteral(true)),
                then_branch: Box::new(Stmt::ExprStmt(Box::new(Expr::IntLiteral(1)))),
                else_branch: Some(Box::new(Stmt::ExprStmt(Box::new(Expr::IntLiteral(2))))),
            },
        ]);

        assert!(validator.validate_program(&program).is_ok());
    }

    #[test]
    fn test_invalid_if_condition() {
        let mut validator = AstValidator::new();
        let program = create_test_program(vec![
            Stmt::If {
                condition: Box::new(Expr::IntLiteral(1)), // Invalid condition type
                then_branch: Box::new(Stmt::ExprStmt(Box::new(Expr::IntLiteral(1)))),
                else_branch: None,
            },
        ]);

        assert!(validator.validate_program(&program).is_err());
    }

    #[test]
    fn test_valid_variable_assignment() {
        let mut validator = AstValidator::new();
        let program = create_test_program(vec![
            Stmt::VarDecl {
                name: "x".to_string(),
                var_type: AstType::Int,
                init_expr: Some(Box::new(Expr::IntLiteral(42))),
            },
            Stmt::VarAssign {
                name: "x".to_string(),
                expr: Box::new(Expr::IntLiteral(10)),
            },
        ]);

        assert!(validator.validate_program(&program).is_ok());
    }

    #[test]
    fn test_invalid_variable_assignment() {
        let mut validator = AstValidator::new();
        let program = create_test_program(vec![
            Stmt::VarDecl {
                name: "x".to_string(),
                var_type: AstType::Int,
                init_expr: Some(Box::new(Expr::IntLiteral(42))),
            },
            Stmt::VarAssign {
                name: "x".to_string(),
                expr: Box::new(Expr::BoolLiteral(true)), // Type mismatch
            },
        ]);

        assert!(validator.validate_program(&program).is_err());
    }
}
