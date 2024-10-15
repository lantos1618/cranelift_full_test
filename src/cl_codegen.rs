// src/cl_codegen.rs

use cranelift::prelude::*;
use cranelift_module::{FuncId, FuncOrDataId, Linkage, Module};

use crate::ast::*;

pub struct CodeGenerator<'a, M: Module> {
    pub module: &'a mut M,
}

impl<'a, M: Module> CodeGenerator<'a, M> {
    pub fn new(module: &'a mut M) -> Self {
        Self { module }
    }

    pub fn compile_function(&mut self, func: &AstFunction) -> FuncId {
        // Create a signature for the function
        let mut signature = self.module.make_signature();

        // Handle return type
        if let AstType::Void = func.return_type {
            // No return type
        } else {
            signature
                .returns
                .push(AbiParam::new(self.translate_type(&func.return_type)));
        }

        // Handle parameters
        for param in &func.params {
            signature
                .params
                .push(AbiParam::new(self.translate_type(&param.ty)));
        }

        // Declare the function
        let func_id = self
            .module
            .declare_function(&func.name, Linkage::Export, &signature)
            .unwrap();

        // Create a function context
        let mut ctx = self.module.make_context();
        ctx.func.signature = signature;

        let mut builder_context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_context);

        // Entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        // Map parameters to variables
        let mut variables = VariableMap::new();
        for (i, param) in func.params.iter().enumerate() {
            let param_value = builder.block_params(entry_block)[i];
            let var = Variable::new(variables.len());
            builder.declare_var(var, self.translate_type(&param.ty));
            builder.def_var(var, param_value);
            variables.insert(param.name.clone(), var);
        }

        // Compile function body
        let block_terminated = self.compile_statements(&func.body, &mut builder, &mut variables);

        // If the block is not terminated, insert a return instruction
        if !block_terminated {
            if let AstType::Void = func.return_type {
                builder.ins().return_(&[]);
            } else {
                let default_value = builder.ins().iconst(self.translate_type(&func.return_type), 0);
                builder.ins().return_(&[default_value]);
            }
        }

        // Finalize the function
        builder.seal_all_blocks();
        builder.finalize();

        // Define the function in the module
        self.module.define_function(func_id, &mut ctx).unwrap();

        func_id
    }

    fn translate_type(&self, ty: &AstType) -> types::Type {
        match ty {
            AstType::Int => types::I64,
            AstType::Float => types::F64,
            AstType::Bool => types::I8, // Represent booleans as 8-bit integers
            AstType::Char => types::I8,
            AstType::Array(_, _) => types::I64, // Arrays as pointers
            AstType::Custom(_) => types::I64,   // Custom types as pointers
            AstType::Void => types::INVALID,    // Void type
        }
    }

    fn compile_statements(
        &mut self,
        statements: &[AstStmt],
        builder: &mut FunctionBuilder,
        variables: &mut VariableMap,
    ) -> bool {
        let mut block_terminated = false;
        for stmt in statements {
            if block_terminated {
                break;
            }
            match stmt {
                AstStmt::VariableDeclaration { name, ty, value } => {
                    let var = Variable::new(variables.len());
                    builder.declare_var(var, self.translate_type(ty));
                    if let Some(expr) = value {
                        let val = self.compile_expression(expr, builder, variables);
                        builder.def_var(var, val);
                    }
                    variables.insert(name.clone(), var);
                }
                AstStmt::Assignment { name, value } => {
                    let val = self.compile_expression(value, builder, variables);
                    if let Some(var) = variables.get(name) {
                        builder.def_var(*var, val);
                    } else {
                        panic!("Undefined variable {}", name);
                    }
                }
                AstStmt::Return(expr) => {
                    let ret_val = self.compile_expression(expr, builder, variables);
                    builder.ins().return_(&[ret_val]);
                    block_terminated = true;
                }
                AstStmt::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    let terminated = self.compile_if_statement(
                        condition,
                        then_branch,
                        else_branch,
                        builder,
                        variables,
                    );
                    if terminated {
                        block_terminated = true;
                    }
                }
                AstStmt::While { condition, body } => {
                    self.compile_while_statement(condition, body, builder, variables);
                    // Loops don't terminate the block
                }
                AstStmt::Expression(expr) => {
                    self.compile_expression(expr, builder, variables);
                }
                _ => unimplemented!("Statement type not implemented"),
            }
        }
        block_terminated
    }

    fn compile_expression(
        &mut self,
        expr: &AstExpr,
        builder: &mut FunctionBuilder,
        variables: &VariableMap,
    ) -> Value {
        match expr {
            AstExpr::Literal(lit) => self.compile_literal(lit, builder),
            AstExpr::Variable(name) => {
                if let Some(var) = variables.get(name) {
                    builder.use_var(*var)
                } else {
                    panic!("Undefined variable {}", name);
                }
            }
            AstExpr::BinaryOperation {
                left,
                operator,
                right,
            } => {
                let left_val = self.compile_expression(left, builder, variables);
                let right_val = self.compile_expression(right, builder, variables);
                self.compile_binary_operation(*operator, left_val, right_val, builder)
            }
            AstExpr::UnaryOperation { operator, operand } => {
                let operand_val = self.compile_expression(operand, builder, variables);
                self.compile_unary_operation(*operator, operand_val, builder)
            }
            AstExpr::FunctionCall { name, args } => {
                self.compile_function_call(name, args, builder, variables)
            }
            _ => unimplemented!("Expression type not implemented"),
        }
    }

    fn compile_literal(&self, lit: &Literal, builder: &mut FunctionBuilder) -> Value {
        match lit {
            Literal::Int(i) => builder.ins().iconst(types::I64, *i),
            Literal::Float(f) => builder.ins().f64const(*f),
            Literal::Bool(b) => builder.ins().iconst(types::I8, if *b { 1 } else { 0 }),
            Literal::Char(c) => builder.ins().iconst(types::I8, *c as i64),
            Literal::String(_s) => unimplemented!("String literals are not implemented"),
        }
    }

    fn compile_binary_operation(
        &self,
        operator: AstOperator,
        left: Value,
        right: Value,
        builder: &mut FunctionBuilder,
    ) -> Value {
        match operator {
            // Arithmetic operations
            AstOperator::Add => builder.ins().iadd(left, right),
            AstOperator::Subtract => builder.ins().isub(left, right),
            AstOperator::Multiply => builder.ins().imul(left, right),
            AstOperator::Divide => builder.ins().sdiv(left, right),
            AstOperator::Modulo => builder.ins().srem(left, right),
            // Comparison operations
            AstOperator::Equal => {
                let cmp = builder.ins().icmp(IntCC::Equal, left, right);
                builder.ins().uextend(types::I64, cmp)
            }
            AstOperator::NotEqual => {
                let cmp = builder.ins().icmp(IntCC::NotEqual, left, right);
                builder.ins().uextend(types::I64, cmp)
            }
            AstOperator::Less => {
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, left, right);
                builder.ins().uextend(types::I64, cmp)
            }
            AstOperator::Greater => {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, left, right);
                builder.ins().uextend(types::I64, cmp)
            }
            AstOperator::LessEqual => {
                let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, left, right);
                builder.ins().uextend(types::I64, cmp)
            }
            AstOperator::GreaterEqual => {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, left, right);
                builder.ins().uextend(types::I64, cmp)
            }
            // Logical operations
            AstOperator::And => {
                // Ensure operands are booleans
                let left_bool = builder.ins().ireduce(types::I8, left);
                let right_bool = builder.ins().ireduce(types::I8, right);
                let result = builder.ins().band(left_bool, right_bool);
                builder.ins().uextend(types::I64, result)
            }
            AstOperator::Or => {
                // Ensure operands are booleans
                let left_bool = builder.ins().ireduce(types::I8, left);
                let right_bool = builder.ins().ireduce(types::I8, right);
                let result = builder.ins().bor(left_bool, right_bool);
                builder.ins().uextend(types::I64, result)
            }
            // Bitwise operations
            AstOperator::BitwiseAnd => builder.ins().band(left, right),
            AstOperator::BitwiseOr => builder.ins().bor(left, right),
            AstOperator::BitwiseXor => builder.ins().bxor(left, right),
            // Shift operations
            AstOperator::ShiftLeft => builder.ins().ishl(left, right),
            AstOperator::ShiftRight => builder.ins().sshr(left, right),
            _ => unimplemented!("Operator {:?} not implemented", operator),
        }
    }

    fn compile_unary_operation(
        &self,
        operator: UnaryOperator,
        operand: Value,
        builder: &mut FunctionBuilder,
    ) -> Value {
        match operator {
            UnaryOperator::Negate => builder.ins().ineg(operand),
            UnaryOperator::Not => {
                let zero = builder.ins().iconst(types::I64, 0);
                let cmp = builder.ins().icmp(IntCC::Equal, operand, zero);
                builder.ins().uextend(types::I64, cmp)
            }
            UnaryOperator::BitwiseNot => builder.ins().bnot(operand),
        }
    }

    fn compile_function_call(
        &mut self,
        name: &str,
        args: &[AstExpr],
        builder: &mut FunctionBuilder,
        variables: &VariableMap,
    ) -> Value {
        // Retrieve or declare the function reference
        let func_ref = if let Some(id) = self.module.get_name(name) {
            id
        } else {
            // For simplicity, assume the function has the same signature
            let mut sig = self.module.make_signature();
            // For this example, all parameters and return types are I64
            for _ in args {
                sig.params.push(AbiParam::new(types::I64));
            }
            sig.returns.push(AbiParam::new(types::I64));
            let func_id = self
                .module
                .declare_function(name, Linkage::Export, &sig)
                .unwrap();
            FuncOrDataId::Func(func_id)
        };

        // Ensure the function is declared in the current function
        let func_id = match func_ref {
            FuncOrDataId::Func(id) => id,
            _ => panic!("Expected a function ID"),
        };
        let local_callee = self.module.declare_func_in_func(func_id, builder.func);

        let mut arg_values = Vec::new();
        for arg in args {
            let arg_val = self.compile_expression(arg, builder, variables);
            arg_values.push(arg_val);
        }

        let call = builder.ins().call(local_callee, &arg_values);

        if self.translate_type(&AstType::Int) == types::INVALID {
            // Function has no return value
            builder.ins().iconst(types::I64, 0) // Return a dummy value
        } else {
            builder.inst_results(call)[0]
        }
    }

    fn compile_if_statement(
        &mut self,
        condition: &AstExpr,
        then_branch: &[AstStmt],
        else_branch: &Option<Vec<AstStmt>>,
        builder: &mut FunctionBuilder,
        variables: &mut VariableMap,
    ) -> bool {
        let then_block = builder.create_block();
        let else_block = builder.create_block();
        let merge_block = builder.create_block();

        // Evaluate the condition
        let condition_val = self.compile_expression(condition, builder, variables);

        // Use 'brif' with the condition value
        builder
            .ins()
            .brif(condition_val, then_block, &[], else_block, &[]);

        // Then block
        builder.switch_to_block(then_block);
        let then_terminated = self.compile_statements(then_branch, builder, variables);
        if !then_terminated {
            builder.ins().jump(merge_block, &[]);
        }
        builder.seal_block(then_block);

        // Else block
        builder.switch_to_block(else_block);
        let else_terminated = if let Some(else_stmts) = else_branch {
            self.compile_statements(else_stmts, builder, variables)
        } else {
            false // No else statements
        };
        if !else_terminated {
            builder.ins().jump(merge_block, &[]);
        }
        builder.seal_block(else_block);

        // Merge block
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        // The block is terminated only if both branches are terminated
        then_terminated && else_terminated
    }

    fn compile_while_statement(
        &mut self,
        condition: &AstExpr,
        body: &[AstStmt],
        builder: &mut FunctionBuilder,
        variables: &mut VariableMap,
    ) {
        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let after_loop = builder.create_block();

        // Initial jump to loop header
        builder.ins().jump(loop_header, &[]);

        // Loop header block
        builder.switch_to_block(loop_header);

        // Evaluate the condition
        let condition_val = self.compile_expression(condition, builder, variables);

        // Use 'brif' with the condition value
        builder
            .ins()
            .brif(condition_val, loop_body, &[], after_loop, &[]);

        // Loop body block
        builder.switch_to_block(loop_body);
        let body_terminated = self.compile_statements(body, builder, variables);
        if !body_terminated {
            builder.ins().jump(loop_header, &[]);
        }
        builder.seal_block(loop_body);

        // Now we can seal loop_header after the back edge has been added
        builder.seal_block(loop_header);

        // After loop block
        builder.switch_to_block(after_loop);
        // Do not seal after_loop here; it may have successors later
    }
}
