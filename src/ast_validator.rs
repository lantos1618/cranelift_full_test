use crate::ast::{Program, Stmt, Expr, BinOp, AstType, FuncDecl, UnaryOp};
use crate::error::{CompilerError, ErrorType};
use std::collections::HashMap;

pub struct AstValidator {
    variables: HashMap<String, AstType>,
    functions: HashMap<String, (Vec<AstType>, AstType)>,
}

impl AstValidator {
    pub fn new() -> Self {
        AstValidator {
            variables: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    pub fn validate_program(&mut self, program: &Program) -> Result<(), CompilerError> {
        for stmt in &program.statements {
            self.validate_stmt(stmt)?;
        }
        Ok(())
    }

    fn validate_stmt(&mut self, stmt: &Stmt) -> Result<(), CompilerError> {
        match stmt {
            Stmt::ExprStmt(expr) => {
                self.validate_expr(expr)?;
            }
            Stmt::Return(expr) => {
                self.validate_expr(expr)?;
            }
            Stmt::VarAssign { name, expr } => {
                let expr_type = self.validate_expr(expr)?;
                self.variables.insert(name.clone(), expr_type);
            }
            Stmt::VarDecl { name, var_type, init_expr } => {
                if let Some(init_expr) = init_expr {
                    let expr_type = self.validate_expr(init_expr)?;
                    if expr_type != *var_type {
                        return Err(CompilerError::new(
                            format!("Type mismatch in variable declaration: expected {:?}, found {:?}", var_type, expr_type),
                            0, 0, "".to_string(), ErrorType::Semantic,
                        ));
                    }
                }
                self.variables.insert(name.clone(), var_type.clone());
            }
            Stmt::Block(stmts) => {
                let old_vars = self.variables.clone();
                for stmt in stmts {
                    self.validate_stmt(stmt)?;
                }
                self.variables = old_vars;
            }
            Stmt::FuncDef { func_decl, body } => {
                self.validate_func_def(func_decl, body)?;
            }
            Stmt::StructDef { name: _, fields: _ } => {
                // Struct definitions are always valid
            }
            Stmt::If { condition, then_branch, else_branch } => {
                let cond_type = self.validate_expr(condition)?;
                if cond_type != AstType::Bool {
                    return Err(CompilerError::new(
                        "If condition must be a boolean".to_string(),
                        0, 0, "".to_string(), ErrorType::Semantic,
                    ));
                }
                self.validate_stmt(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.validate_stmt(else_branch)?;
                }
            }
            Stmt::While { condition, body } => {
                let cond_type = self.validate_expr(condition)?;
                if cond_type != AstType::Bool {
                    return Err(CompilerError::new(
                        "While condition must be a boolean".to_string(),
                        0, 0, "".to_string(), ErrorType::Semantic,
                    ));
                }
                self.validate_stmt(body)?;
            }
            Stmt::Break | Stmt::Continue => {
                // These are valid in loops
            }
            Stmt::FuncExternDecl { name: _, lib: _, .. } => {
                // External function declarations are always valid
            }
        }
        Ok(())
    }

    fn validate_expr(&mut self, expr: &Expr) -> Result<AstType, CompilerError> {
        match expr {
            Expr::IntLiteral(_) => Ok(AstType::Int),
            Expr::BoolLiteral(_) => Ok(AstType::Bool),
            Expr::StringLiteral(_) => Ok(AstType::String),
            Expr::CharLiteral(_) => Ok(AstType::Char),
            Expr::Variable(name) => {
                self.variables.get(name).cloned().ok_or_else(|| {
                    CompilerError::new(
                        format!("Undefined variable: {}", name),
                        0, 0, "".to_string(), ErrorType::Semantic,
                    )
                })
            }
            Expr::BinaryOp(left, op, right) => {
                let left_type = self.validate_expr(left)?;
                let right_type = self.validate_expr(right)?;

                match op {
                    BinOp::Add | BinOp::Subtract | BinOp::Multiply | BinOp::Divide | BinOp::Modulus => {
                        if left_type != AstType::Int || right_type != AstType::Int {
                            return Err(CompilerError::new(
                                format!("Invalid types for arithmetic operation: found {:?} and {:?}, expected Int", left_type, right_type),
                                0, 0, "".to_string(), ErrorType::Semantic,
                            ));
                        }
                        Ok(AstType::Int)
                    }
                    BinOp::And | BinOp::Or => {
                        if left_type != AstType::Bool || right_type != AstType::Bool {
                            return Err(CompilerError::new(
                                "Logical operations require boolean operands".to_string(),
                                0, 0, "".to_string(), ErrorType::Semantic,
                            ));
                        }
                        Ok(AstType::Bool)
                    }
                    BinOp::Equal | BinOp::NotEqual | BinOp::LessThan | BinOp::GreaterThan => {
                        if left_type != right_type {
                            return Err(CompilerError::new(
                                "Comparison requires same type operands".to_string(),
                                0, 0, "".to_string(), ErrorType::Semantic,
                            ));
                        }
                        Ok(AstType::Bool)
                    }
                    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::ShiftLeft | BinOp::ShiftRight => {
                        if left_type != AstType::Int || right_type != AstType::Int {
                            return Err(CompilerError::new(
                                "Bitwise operations require integer operands".to_string(),
                                0, 0, "".to_string(), ErrorType::Semantic,
                            ));
                        }
                        Ok(AstType::Int)
                    }
                }
            }
            Expr::UnaryOp(op, expr) => {
                let expr_type = self.validate_expr(expr)?;
                match op {
                    UnaryOp::Not => {
                        if expr_type != AstType::Bool {
                            return Err(CompilerError::new(
                                "Logical not requires boolean operand".to_string(),
                                0, 0, "".to_string(), ErrorType::Semantic,
                            ));
                        }
                        Ok(AstType::Bool)
                    }
                    UnaryOp::Negate => {
                        if expr_type != AstType::Int {
                            return Err(CompilerError::new(
                                "Negation requires integer operand".to_string(),
                                0, 0, "".to_string(), ErrorType::Semantic,
                            ));
                        }
                        Ok(AstType::Int)
                    }
                    UnaryOp::Deref => {
                        // Assume deref returns the base type
                        Ok(expr_type)
                    }
                    UnaryOp::AddressOf => {
                        // Assume address-of returns a pointer type
                        Ok(AstType::Pointer(Box::new(expr_type)))
                    }
                    UnaryOp::BitNot => {
                        if expr_type != AstType::Int {
                            return Err(CompilerError::new(
                                "Bitwise not requires integer operand".to_string(),
                                0, 0, "".to_string(), ErrorType::Semantic,
                            ));
                        }
                        Ok(AstType::Int)
                    }
                }
            }
            Expr::FuncCall(name, args) => {
                let saved_vars = self.variables.clone();
                let result = if let Some((param_types, return_type)) = self.functions.get(name).cloned() {
                    if args.len() != param_types.len() {
                        return Err(CompilerError::new(
                            format!("Wrong number of arguments for function {}", name),
                            0, 0, "".to_string(), ErrorType::Semantic,
                        ));
                    }
                    
                    for (arg, expected_type) in args.iter().zip(param_types.iter()) {
                        let arg_type = self.validate_expr(arg)?;
                        if arg_type != *expected_type {
                            return Err(CompilerError::new(
                                format!("Type mismatch in function call: expected {:?}, found {:?}", expected_type, arg_type),
                                0, 0, "".to_string(), ErrorType::Semantic,
                            ));
                        }
                    }
                    
                    Ok(return_type)
                } else {
                    Err(CompilerError::new(
                        format!("Undefined function: {}", name),
                        0, 0, "".to_string(), ErrorType::Semantic,
                    ))
                };
                self.variables = saved_vars;
                result
            }
            Expr::Assignment(target, value) => {
                let target_type = self.validate_expr(target)?;
                let value_type = self.validate_expr(value)?;
                if target_type != value_type {
                    return Err(CompilerError::new(
                        format!("Type mismatch in assignment: expected {:?}, found {:?}", target_type, value_type),
                        0, 0, "".to_string(), ErrorType::Semantic,
                    ));
                }
                Ok(target_type)
            }
            Expr::ArrayAccess { array, index } => {
                let array_type = self.validate_expr(array)?;
                let index_type = self.validate_expr(index)?;
                
                if index_type != AstType::Int {
                    return Err(CompilerError::new(
                        "Array index must be an integer".to_string(),
                        0, 0, "".to_string(), ErrorType::Semantic,
                    ));
                }
                
                match array_type {
                    AstType::Array(elem_type) => Ok(*elem_type),
                    _ => Err(CompilerError::new(
                        "Cannot index into non-array type".to_string(),
                        0, 0, "".to_string(), ErrorType::Semantic,
                    )),
                }
            }
            Expr::ArrayAssignment { array, index, value } => {
                let array_type = self.validate_expr(array)?;
                let index_type = self.validate_expr(index)?;
                let value_type = self.validate_expr(value)?;
                
                if index_type != AstType::Int {
                    return Err(CompilerError::new(
                        "Array index must be an integer".to_string(),
                        0, 0, "".to_string(), ErrorType::Semantic,
                    ));
                }
                
                match array_type {
                    AstType::Array(elem_type) => {
                        if value_type != *elem_type {
                            return Err(CompilerError::new(
                                format!("Type mismatch in array assignment: expected {:?}, found {:?}", elem_type, value_type),
                                0, 0, "".to_string(), ErrorType::Semantic,
                            ));
                        }
                        Ok(value_type)
                    }
                    _ => Err(CompilerError::new(
                        "Cannot index into non-array type".to_string(),
                        0, 0, "".to_string(), ErrorType::Semantic,
                    )),
                }
            }
            Expr::StructAccess(expr, _) => {
                self.validate_expr(expr)?;
                Ok(AstType::Int) // Placeholder
            }
            Expr::StructInit { .. } | Expr::ExternCall { .. } | Expr::Match { .. } => {
                // TODO: Implement validation for these expressions
                Ok(AstType::Int) // Placeholder
            }
        }
    }

    fn validate_func_def(&mut self, func_decl: &FuncDecl, body: &Stmt) -> Result<(), CompilerError> {
        // Save old scope
        let old_vars = self.variables.clone();
        
        // Add parameters to scope
        for (name, typ) in &func_decl.params {
            self.variables.insert(name.clone(), typ.clone());
        }
        
        // Add function to function table
        let param_types: Vec<AstType> = func_decl.params.iter().map(|(_, t)| t.clone()).collect();
        self.functions.insert(func_decl.name.clone(), (param_types, func_decl.return_type.clone()));
        
        // Validate body
        self.validate_stmt(body)?;
        
        // Restore old scope
        self.variables = old_vars;
        
        Ok(())
    }

    fn validate_func_call(&mut self, name: &str, args: &[Expr]) -> Result<AstType, CompilerError> {
        let (param_types, return_type) = self.functions.get(name)
            .cloned()
            .ok_or_else(|| CompilerError::new(
                format!("Undefined function: {}", name),
                0, 0, "".to_string(), ErrorType::Semantic,
            ))?;

        if args.len() != param_types.len() {
            return Err(CompilerError::new(
                format!("Wrong number of arguments for function {}", name),
                0, 0, "".to_string(), ErrorType::Semantic,
            ));
        }
        
        // Create a copy of the variables before validation
        let saved_vars = self.variables.clone();
        
        // Validate all arguments
        for (arg, expected_type) in args.iter().zip(param_types.iter()) {
            let arg_type = self.validate_expr(arg)?;
            if arg_type != *expected_type {
                // Restore variables before returning error
                self.variables = saved_vars;
                return Err(CompilerError::new(
                    format!("Type mismatch in function call: expected {:?}, found {:?}", expected_type, arg_type),
                    0, 0, "".to_string(), ErrorType::Semantic,
                ));
            }
        }
        
        // Restore variables after validation
        self.variables = saved_vars;
        Ok(return_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Stmt, Program, BinOp, AstType, FuncDecl};

    #[test]
    fn test_validate_arithmetic() {
        let mut validator = AstValidator::new();
        let expr = Expr::BinaryOp(
            Box::new(Expr::IntLiteral(1)),
            BinOp::Add,
            Box::new(Expr::IntLiteral(2)),
        );
        assert!(validator.validate_expr(&expr).is_ok());
    }

    #[test]
    fn test_validate_invalid_arithmetic() {
        let mut validator = AstValidator::new();
        let expr = Expr::BinaryOp(
            Box::new(Expr::BoolLiteral(true)),
            BinOp::Add,
            Box::new(Expr::IntLiteral(2)),
        );
        assert!(validator.validate_expr(&expr).is_err());
    }

    #[test]
    fn test_validate_function_call() {
        let mut validator = AstValidator::new();
        
        // Add function to validator
        validator.functions.insert(
            "add".to_string(),
            (vec![AstType::Int, AstType::Int], AstType::Int),
        );
        
        let call = Expr::FuncCall(
            "add".to_string(),
            vec![
                Expr::IntLiteral(1),
                Expr::IntLiteral(2),
            ],
        );
        
        assert!(validator.validate_expr(&call).is_ok());
    }

    #[test]
    fn test_validate_invalid_function_call() {
        let mut validator = AstValidator::new();
        
        // Add function to validator
        validator.functions.insert(
            "add".to_string(),
            (vec![AstType::Int, AstType::Int], AstType::Int),
        );
        
        let call = Expr::FuncCall(
            "add".to_string(),
            vec![
                Expr::IntLiteral(1),
                Expr::BoolLiteral(true),
            ],
        );
        
        assert!(validator.validate_expr(&call).is_err());
    }
}
