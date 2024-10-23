use anyhow::Result;

use cranelift::prelude::*;
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_jit::{JITBuilder, JITModule};
// Removed unused imports

use crate::ast::*;

use std::collections::HashMap;
use std::mem::transmute;


// Define LoopContext struct
struct LoopContext {
    header_block: Block,
    exit_block: Block,
}


pub struct CodeGenerator<'a> {
    pub module: &'a mut JITModule,
    builder_context: FunctionBuilderContext,
    struct_types: HashMap<String, Vec<(String, AstType)>>, // Struct definitions
    function_ids: HashMap<String, FuncId>, // Map function names to FuncId
}

impl<'a> CodeGenerator<'a> {
    pub fn new(module: &'a mut JITModule) -> Self {
        Self {
            module,
            builder_context: FunctionBuilderContext::new(),
            struct_types: HashMap::new(),
            function_ids: HashMap::new(),
        }
    }

    pub fn compile_program(&mut self, program: &Program) -> Result<()> {
        // Create function signature for main
        let mut sig = self.module.make_signature();
        sig.returns.push(AbiParam::new(types::I64)); // Assuming main returns an i64
        let func_id = self.module.declare_function("main", Linkage::Export, &sig)?;

        // Create function context
        let mut func_ctx = FunctionBuilderContext::new();
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;

        // Create a local variables map for the main function
        let mut variables = HashMap::new();

        let mut loop_stack = Vec::new(); // Initialize the loop context stack

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

            let entry_block = builder.create_block();
            builder.switch_to_block(entry_block);
            // Append function parameters to the block (none in this case)

            for stmt in &program.statements {
                self.compile_stmt(stmt, &mut builder, &mut variables, &mut loop_stack)?;

                if self.is_current_block_terminated(&builder) {
                    // The block is terminated, no need to process further statements
                    break;
                }
            }

            // Ensure the function has a return instruction if not already terminated
            if !self.is_current_block_terminated(&builder) {
                // Return 0 by default if no return statement is encountered
                let zero = builder.ins().iconst(types::I64, 0);
                builder.ins().return_(&[zero]);
            }

            // Seal all blocks after defining them
            builder.seal_all_blocks();

            builder.finalize();
        }

        self.module.define_function(func_id, &mut ctx)?;

        // Store the function ID
        self.function_ids.insert("main".to_string(), func_id);

        Ok(())
    }


    pub fn compile_stmt(
        &mut self,
        stmt: &Stmt,
        builder: &mut FunctionBuilder,
        variables: &mut HashMap<String, Variable>,
        loop_stack: &mut Vec<LoopContext>,
    ) -> Result<()> {
        if self.is_current_block_terminated(builder) {
            return Ok(());
        }

        match stmt {
            Stmt::VarDecl {
                name,
                var_type,
                init_expr,
            } => {
                self.compile_var_decl(name, var_type, init_expr.as_deref(), builder, variables)?;
            }
            Stmt::VarAssign { name, expr } => {
                self.compile_var_assign(name, expr, builder, variables)?;
            }
            Stmt::ExprStmt(expr) => {
                self.compile_expr(expr, Some(builder), variables)?;
            }
            Stmt::Return(expr) => {
                self.compile_return(expr, builder, variables)?;
                // After a return, the block is terminated.
                return Ok(());
            }
            Stmt::Block(stmts) => {
                for stmt in stmts {
                    self.compile_stmt(stmt, builder, variables, loop_stack)?;

                    if self.is_current_block_terminated(builder) {
                        break;
                    }
                }
            }
            Stmt::Break => {
                if let Some(loop_context) = loop_stack.last() {
                    builder.ins().jump(loop_context.exit_block, &[]);
                    // Since we've added a jump, the current block is terminated
                    // builder.seal_block(builder.current_block().unwrap()); // Remove this line
                } else {
                    return Err(anyhow::anyhow!("'break' used outside of a loop"));
                }
            }
            Stmt::Continue => {
                if let Some(loop_context) = loop_stack.last() {
                    builder.ins().jump(loop_context.header_block, &[]);
                    // Since we've added a jump, the current block is terminated
                    // builder.seal_block(builder.current_block().unwrap()); // Remove this line
                } else {
                    return Err(anyhow::anyhow!("'continue' used outside of a loop"));
                }
            }
            Stmt::FuncDef { func_decl, body } => {
                self.compile_func_def(func_decl, body)?;
            }
            Stmt::FuncExternDecl { func_decl, lib: _ } => {
                self.compile_func_extern_decl(func_decl)?;
            }
            Stmt::StructDef { name, fields } => {
                self.struct_types.insert(name.clone(), fields.clone());
            }
            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.compile_if_statement(
                    condition,
                    then_branch,
                    else_branch.as_deref(),
                    builder,
                    variables,
                    loop_stack,
                )?;
            }
            Stmt::While { condition, body } => {
                self.compile_while_loop(condition, body, builder, variables, loop_stack)?;
            }
        }
        Ok(())
    }

    pub fn compile_var_decl(
        &mut self,
        name: &str,
        var_type: &AstType,
        init_expr: Option<&Expr>,
        builder: &mut FunctionBuilder,
        variables: &mut HashMap<String, Variable>,
    ) -> Result<()> {
        let var = Variable::new(variables.len());
        let cl_type = self.ast_type_to_cl_type(var_type)?;

        builder.declare_var(var, cl_type);
        variables.insert(name.to_string(), var);

        if let Some(expr) = init_expr {
            let value = self.compile_expr(expr, Some(builder), variables)?;
            builder.def_var(var, value);
        } else {
            // Initialize variable with zero
            let zero = builder.ins().iconst(cl_type, 0);
            builder.def_var(var, zero);
        }

        Ok(())
    }

    pub fn compile_var_assign(
        &mut self,
        name: &str,
        expr: &Expr,
        builder: &mut FunctionBuilder,
        variables: &mut HashMap<String, Variable>,
    ) -> Result<()> {
        if let Some(&var) = variables.get(name) {
            let value = self.compile_expr(expr, Some(builder), variables)?;
            builder.def_var(var, value);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Undefined variable `{}`", name))
        }
    }

    pub fn compile_return(
        &mut self,
        expr: &Expr,
        builder: &mut FunctionBuilder,
        variables: &mut HashMap<String, Variable>,
    ) -> Result<()> {
        let value = self.compile_expr(expr, Some(builder), variables)?;
        builder.ins().return_(&[value]);
        Ok(())
    }

    pub fn compile_func_def(&mut self, func_decl: &FuncDecl, body: &Stmt) -> Result<()> {
        // Create a local variables map for this function
        let mut variables = HashMap::new();

        // Create function signature
        let sig = self.create_signature(&func_decl.params, &func_decl.return_type)?;
        let func_id = self
            .module
            .declare_function(&func_decl.name, Linkage::Local, &sig)?;

        // Create function context
        let mut func_ctx = FunctionBuilderContext::new();
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;

        let mut loop_stack = Vec::new(); // Initialize the loop context stack

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

            let entry_block = builder.create_block();
            builder.switch_to_block(entry_block);

            // Append function parameters to the block
            builder.append_block_params_for_function_params(entry_block);

            // Map function parameters to variables
            for (i, (name, param_type)) in func_decl.params.iter().enumerate() {
                let var = Variable::new(variables.len());
                variables.insert(name.clone(), var);

                let cl_type = self.ast_type_to_cl_type(param_type)?;
                builder.declare_var(var, cl_type);
                let val = builder.block_params(entry_block)[i];
                builder.def_var(var, val);
            }

            self.compile_stmt(body, &mut builder, &mut variables, &mut loop_stack)?;

            // Ensure the function has a return instruction if not already terminated
            if !self.is_current_block_terminated(&builder) {
                if let AstType::Void = func_decl.return_type {
                    builder.ins().return_(&[]);
                } else {
                    // Return zero if no return statement is present
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().return_(&[zero]);
                }
            }

            // Seal all blocks after defining them
            builder.seal_all_blocks();

            builder.finalize();
        }

        self.module.define_function(func_id, &mut ctx)?;

        // Store the function ID
        self.function_ids.insert(func_decl.name.clone(), func_id);

        Ok(())
    }

    pub fn compile_func_extern_decl(&mut self, func_decl: &FuncDecl) -> Result<()> {
        let sig = self.create_signature(&func_decl.params, &func_decl.return_type)?;
        self.module
            .declare_function(&func_decl.name, Linkage::Import, &sig)?;
        Ok(())
    }

    pub fn compile_expr(
        &mut self,
        expr: &Expr,
        builder_opt: Option<&mut FunctionBuilder>,
        variables: &mut HashMap<String, Variable>,
    ) -> Result<Value> {
        if let Some(builder) = builder_opt {
            if self.is_current_block_terminated(builder) {
                return Err(anyhow::anyhow!("Cannot add instructions to a terminated block"));
            }

            match expr {
                Expr::IntLiteral(val) => Ok(builder.ins().iconst(types::I64, *val)),
                Expr::BoolLiteral(val) => {
                    // Booleans represented as I8 (1-byte integer)
                    let bool_value = if *val { 1 } else { 0 };
                    Ok(builder.ins().iconst(types::I8, bool_value))
                }
                Expr::Variable(name) => {
                    if let Some(&var) = variables.get(name) {
                        Ok(builder.use_var(var))
                    } else {
                        Err(anyhow::anyhow!("Undefined variable `{}`", name))
                    }
                }
                Expr::BinaryOp(lhs, op, rhs) => {
                    let lhs_val = self.compile_expr(lhs, Some(builder), variables)?;
                    let rhs_val = self.compile_expr(rhs, Some(builder), variables)?;
                    let result = match op {
                        BinOp::Add => builder.ins().iadd(lhs_val, rhs_val),
                        BinOp::Subtract => builder.ins().isub(lhs_val, rhs_val),
                        BinOp::Multiply => builder.ins().imul(lhs_val, rhs_val),
                        BinOp::Divide => builder.ins().sdiv(lhs_val, rhs_val),
                        BinOp::Modulus => builder.ins().srem(lhs_val, rhs_val), // Implement modulus
                        BinOp::Equal => {
                            let cmp = builder.ins().icmp(IntCC::Equal, lhs_val, rhs_val);
                            builder.ins().bmask(types::I8, cmp)
                        }
                        BinOp::NotEqual => {
                            let cmp = builder.ins().icmp(IntCC::NotEqual, lhs_val, rhs_val);
                            builder.ins().bmask(types::I8, cmp)
                        }
                        BinOp::LessThan => {
                            let cmp = builder.ins().icmp(IntCC::SignedLessThan, lhs_val, rhs_val);
                            builder.ins().bmask(types::I8, cmp)
                        }
                        BinOp::GreaterThan => {
                            let cmp =
                                builder.ins().icmp(IntCC::SignedGreaterThan, lhs_val, rhs_val);
                            builder.ins().bmask(types::I8, cmp)
                        }
                        _ => unimplemented!("Operator {:?} not implemented", op),
                    };
                    Ok(result)
                }
                Expr::UnaryOp(op, expr) => {
                    let val = self.compile_expr(expr, Some(builder), variables)?;
                    let result = match op {
                        UnaryOp::Negate => builder.ins().ineg(val),
                        UnaryOp::Not => {
                            let ty = builder.func.dfg.value_type(val);
                            let zero = builder.ins().iconst(ty, 0);
                            let cmp = builder.ins().icmp(IntCC::Equal, val, zero);
                            builder.ins().bmask(ty, cmp)
                        }
                        _ => unimplemented!("Unary operator {:?} not implemented", op),
                    };
                    Ok(result)
                }
                Expr::FuncCall(name, args) => {
                    if let Some(&func_id) = self.function_ids.get(name) {
                        let func_ref = self.module.declare_func_in_func(func_id, builder.func);
                        let mut arg_values = Vec::new();
                        for arg in args {
                            arg_values.push(self.compile_expr(arg, Some(builder), variables)?);
                        }
                        let call = builder.ins().call(func_ref, &arg_values);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Err(anyhow::anyhow!("Function `{}` has no return value", name))
                        } else {
                            Ok(results[0])
                        }
                    } else {
                        Err(anyhow::anyhow!("Undefined function `{}`", name))
                    }
                }
                _ => unimplemented!("Expression {:?} not implemented", expr),
            }
        } else {
            Err(anyhow::anyhow!("No builder provided in compile_expr"))
        }
    }

    pub fn compile_if_statement(
        &mut self,
        condition: &Expr,
        then_branch: &Stmt,
        else_branch: Option<&Stmt>,
        builder: &mut FunctionBuilder,
        variables: &mut HashMap<String, Variable>,
        loop_stack: &mut Vec<LoopContext>,
    ) -> Result<()> {
        let cond_val = self.compile_expr(condition, Some(builder), variables)?;

        // Since booleans are I8, compare with zero to get a boolean condition
        let zero = builder.ins().iconst(types::I8, 0);
        let cmp = builder.ins().icmp(IntCC::NotEqual, cond_val, zero);

        let then_block = builder.create_block();
        let else_block = builder.create_block();
        let merge_block = builder.create_block();

        // Declare that the current block will transition to then_block and else_block
        builder.ins().brif(cmp, then_block, &[], else_block, &[]);

        // Then block
        builder.switch_to_block(then_block);
        self.compile_stmt(then_branch, builder, variables, loop_stack)?;

        if !self.is_current_block_terminated(builder) {
            builder.ins().jump(merge_block, &[]);
        }

        // Else block
        builder.switch_to_block(else_block);
        if let Some(else_stmt) = else_branch {
            self.compile_stmt(else_stmt, builder, variables, loop_stack)?;
        }

        if !self.is_current_block_terminated(builder) {
            builder.ins().jump(merge_block, &[]);
        }

        // Seal the blocks
        builder.seal_block(then_block);
        builder.seal_block(else_block);

        // Merge block
        builder.switch_to_block(merge_block);
        // Seal the merge block now that all predecessors are known
        builder.seal_block(merge_block);

        Ok(())
    }

    pub fn compile_while_loop(
        &mut self,
        condition: &Expr,
        body: &Stmt,
        builder: &mut FunctionBuilder,
        variables: &mut HashMap<String, Variable>,
        loop_stack: &mut Vec<LoopContext>,
    ) -> Result<()> {
        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_exit = builder.create_block();

        // Push the current loop context onto the stack
        loop_stack.push(LoopContext {
            header_block: loop_header,
            exit_block: loop_exit,
        });

        // Jump to loop header
        builder.ins().jump(loop_header, &[]);

        // Loop header
        builder.switch_to_block(loop_header);
        // Do not seal loop_header yet

        let cond_val = self.compile_expr(condition, Some(builder), variables)?;
        let zero = builder.ins().iconst(types::I8, 0);
        let cmp = builder.ins().icmp(IntCC::NotEqual, cond_val, zero);

        builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

        // Loop body
        builder.switch_to_block(loop_body);
        self.compile_stmt(body, builder, variables, loop_stack)?;

        if !self.is_current_block_terminated(builder) {
            builder.ins().jump(loop_header, &[]);
        }
        // Seal the loop_body
        builder.seal_block(loop_body);

        // Now we can seal the loop_header
        builder.seal_block(loop_header);

        // Pop the loop context
        loop_stack.pop();

        // Loop exit
        builder.switch_to_block(loop_exit);
        // Seal loop_exit
        builder.seal_block(loop_exit);

        Ok(())
    }

    fn create_signature(
        &self,
        params: &[(String, AstType)],
        return_type: &AstType,
    ) -> Result<Signature> {
        let mut sig = self.module.make_signature();

        for (_, param_type) in params {
            let cl_type = self.ast_type_to_cl_type(param_type)?;
            sig.params.push(AbiParam::new(cl_type));
        }

        if let AstType::Void = return_type {
            // Do nothing
        } else {
            let ret_type = self.ast_type_to_cl_type(return_type)?;
            sig.returns.push(AbiParam::new(ret_type));
        }

        Ok(sig)
    }

    fn ast_type_to_cl_type(&self, ast_type: &AstType) -> Result<Type> {
        match ast_type {
            AstType::Int => Ok(types::I64),
            AstType::Bool => Ok(types::I8), // Use I8 for boolean representation
            AstType::Void => Ok(types::INVALID), // Use types::INVALID for void
            _ => Err(anyhow::anyhow!("Type {:?} not implemented", ast_type)),
        }
    }

    pub fn get_function_id(&self, name: &str) -> Option<FuncId> {
        self.function_ids.get(name).cloned()
    }

    fn is_current_block_terminated(&self, builder: &FunctionBuilder) -> bool {
        if let Some(block) = builder.current_block() {
            if let Some(inst) = builder.func.layout.last_inst(block) {
                let inst_data = &builder.func.dfg.insts[inst];
                inst_data.opcode().is_terminator()  // Return this boolean value directly
            } else {
                false 
            }
        } else {
            true
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to run code
    fn run_code<T>(module: &mut JITModule, func_id: FuncId) -> T {
        // Finalize the function definitions
        module.finalize_definitions().unwrap();

        // Get a pointer to the function's code
        let code = module.get_finalized_function(func_id);

        // Cast the code pointer to a callable function
        unsafe { transmute::<_, fn() -> T>(code)() }
    }

    // Your tests go here...


    #[test]
    fn test_integer_literal() {
        // Setup the ISA and JIT module
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            // Create a code generator
            let mut codegen = CodeGenerator::new(&mut module);

            // Build a program that returns an integer literal
            let prog = Program::new(vec![Stmt::Return(Box::new(Expr::IntLiteral(42)))]);

            // Compile the program
            codegen.compile_program(&prog).expect("Compilation failed");

            // Get the function ID
            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_variable_declaration() {
        // Variable declaration and usage
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            let prog = Program::new(vec![
                Stmt::VarDecl {
                    name: "x".to_string(),
                    var_type: AstType::Int,
                    init_expr: Some(Box::new(Expr::IntLiteral(10))),
                },
                Stmt::Return(Box::new(Expr::Variable("x".to_string()))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 10);
    }

    #[test]
    fn test_variable_assignment() {
        // Variable assignment after declaration
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            let prog = Program::new(vec![
                Stmt::VarDecl {
                    name: "y".to_string(),
                    var_type: AstType::Int,
                    init_expr: Some(Box::new(Expr::IntLiteral(5))),
                },
                Stmt::VarAssign {
                    name: "y".to_string(),
                    expr: Box::new(Expr::IntLiteral(15)),
                },
                Stmt::Return(Box::new(Expr::Variable("y".to_string()))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 15);
    }

    #[test]
    fn test_unary_operations() {
        // Test unary negation
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            let prog = Program::new(vec![
                Stmt::Return(Box::new(Expr::UnaryOp(
                    UnaryOp::Negate,
                    Box::new(Expr::IntLiteral(10)),
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, -10);
    }

    #[test]
    fn test_binary_operations() {
        // Test addition
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            let prog = Program::new(vec![
                Stmt::Return(Box::new(Expr::BinaryOp(
                    Box::new(Expr::IntLiteral(10)),
                    BinOp::Add,
                    Box::new(Expr::IntLiteral(20)),
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 30);
    }

    #[test]
    fn test_function_call() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            // Define a function that adds 1
            let func_decl = FuncDecl {
                name: "add_one".to_string(),
                params: vec![("value".to_string(), AstType::Int)],
                return_type: AstType::Int,
            };

            let func_body = Stmt::Block(vec![
                Stmt::Return(Box::new(Expr::BinaryOp(
                    Box::new(Expr::Variable("value".to_string())),
                    BinOp::Add,
                    Box::new(Expr::IntLiteral(1)),
                ))),
            ]);

            let prog = Program::new(vec![
                Stmt::FuncDef {
                    func_decl: func_decl.clone(),
                    body: Box::new(func_body),
                },
                Stmt::Return(Box::new(Expr::FuncCall(
                    "add_one".to_string(),
                    vec![Expr::IntLiteral(41)],
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_if_else_statement() {
        // Test if-else statement
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            let prog = Program::new(vec![
                Stmt::If {
                    condition: Box::new(Expr::BoolLiteral(true)),
                    then_branch: Box::new(Stmt::Return(Box::new(Expr::IntLiteral(1)))),
                    else_branch: Some(Box::new(Stmt::Return(Box::new(Expr::IntLiteral(0))))),
                },
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_while_loop() {
        // Test while loop
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            let prog = Program::new(vec![
                Stmt::VarDecl {
                    name: "i".to_string(),
                    var_type: AstType::Int,
                    init_expr: Some(Box::new(Expr::IntLiteral(0))),
                },
                Stmt::While {
                    condition: Box::new(Expr::BinaryOp(
                        Box::new(Expr::Variable("i".to_string())),
                        BinOp::LessThan,
                        Box::new(Expr::IntLiteral(5)),
                    )),
                    body: Box::new(Stmt::Block(vec![
                        Stmt::VarAssign {
                            name: "i".to_string(),
                            expr: Box::new(Expr::BinaryOp(
                                Box::new(Expr::Variable("i".to_string())),
                                BinOp::Add,
                                Box::new(Expr::IntLiteral(1)),
                            )),
                        },
                    ])),
                },
                Stmt::Return(Box::new(Expr::Variable("i".to_string()))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 5);
    }

    #[test]
    fn test_return_statement() {
        // Test return statement
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            let prog = Program::new(vec![
                Stmt::Return(Box::new(Expr::IntLiteral(100))),
                // This statement should not be executed
                Stmt::Return(Box::new(Expr::IntLiteral(200))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 100);
    }

    #[test]
    fn test_break_continue_statements() {
        // Test break and continue
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            // Create a code generator
            let mut codegen = CodeGenerator::new(&mut module);

            // Build a program that uses break and continue
            let prog = Program::new(vec![
                Stmt::VarDecl {
                    name: "sum".to_string(),
                    var_type: AstType::Int,
                    init_expr: Some(Box::new(Expr::IntLiteral(0))),
                },
                Stmt::VarDecl {
                    name: "i".to_string(),
                    var_type: AstType::Int,
                    init_expr: Some(Box::new(Expr::IntLiteral(0))),
                },
                Stmt::While {
                    condition: Box::new(Expr::BinaryOp(
                        Box::new(Expr::Variable("i".to_string())),
                        BinOp::LessThan,
                        Box::new(Expr::IntLiteral(10)),
                    )),
                    body: Box::new(Stmt::Block(vec![
                        Stmt::VarAssign {
                            name: "i".to_string(),
                            expr: Box::new(Expr::BinaryOp(
                                Box::new(Expr::Variable("i".to_string())),
                                BinOp::Add,
                                Box::new(Expr::IntLiteral(1)),
                            )),
                        },
                        Stmt::If {
                            condition: Box::new(Expr::BinaryOp(
                                Box::new(Expr::BinaryOp(
                                    Box::new(Expr::Variable("i".to_string())),
                                    BinOp::Modulus,
                                    Box::new(Expr::IntLiteral(2)),
                                )),
                                BinOp::Equal,
                                Box::new(Expr::IntLiteral(0)),
                            )),
                            then_branch: Box::new(Stmt::Continue),
                            else_branch: None,
                        },
                        Stmt::If {
                            condition: Box::new(Expr::BinaryOp(
                                Box::new(Expr::Variable("i".to_string())),
                                BinOp::Equal,
                                Box::new(Expr::IntLiteral(7)),
                            )),
                            then_branch: Box::new(Stmt::Break),
                            else_branch: None,
                        },
                        Stmt::VarAssign {
                            name: "sum".to_string(),
                            expr: Box::new(Expr::BinaryOp(
                                Box::new(Expr::Variable("sum".to_string())),
                                BinOp::Add,
                                Box::new(Expr::Variable("i".to_string())),
                            )),
                        },
                    ])),
                },
                Stmt::Return(Box::new(Expr::Variable("sum".to_string()))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 9); // sum of odd numbers less than 7 (1 + 3 + 5)
    }

    #[test]
    fn test_complex_program() {
        // Build a complex program combining multiple features
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            let prog = Program::new(vec![
                // int x = 0;
                Stmt::VarDecl {
                    name: "x".to_string(),
                    var_type: AstType::Int,
                    init_expr: Some(Box::new(Expr::IntLiteral(0))),
                },
                // while (x < 10) { x = x + 1; }
                Stmt::While {
                    condition: Box::new(Expr::BinaryOp(
                        Box::new(Expr::Variable("x".to_string())),
                        BinOp::LessThan,
                        Box::new(Expr::IntLiteral(10)),
                    )),
                    body: Box::new(Stmt::Block(vec![
                        Stmt::VarAssign {
                            name: "x".to_string(),
                            expr: Box::new(Expr::BinaryOp(
                                Box::new(Expr::Variable("x".to_string())),
                                BinOp::Add,
                                Box::new(Expr::IntLiteral(1)),
                            )),
                        },
                    ])),
                },
                // return x;
                Stmt::Return(Box::new(Expr::Variable("x".to_string()))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 10);
    }
}



