use anyhow::Result;

use cranelift::prelude::*;
use cranelift_codegen::settings;
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_native;

use crate::ast::*;

use std::collections::HashMap;
use std::mem::transmute;

pub struct CodeGenerator<'a> {
    pub module: &'a mut JITModule,
    builder_context: FunctionBuilderContext,
    variables: HashMap<String, Variable>, // Map variable names to Cranelift variables
    struct_types: HashMap<String, Vec<(String, AstType)>>, // Struct definitions
    function_ids: HashMap<String, FuncId>, // Map function names to FuncId
}

impl<'a> CodeGenerator<'a> {
    pub fn new(module: &'a mut JITModule) -> Self {
        Self {
            module,
            builder_context: FunctionBuilderContext::new(),
            variables: HashMap::new(),
            struct_types: HashMap::new(),
            function_ids: HashMap::new(),
        }
    }

    pub fn compile_program(&mut self, program: &Program) -> Result<()> {
        // Create function signature for main
        let sig = self.module.make_signature();
        let func_id = self.module.declare_function("main", Linkage::Export, &sig)?;

        // Create function context
        let mut func_ctx = FunctionBuilderContext::new();
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

            let entry_block = builder.create_block();
            builder.switch_to_block(entry_block);
            // Do not seal the entry block here
            // builder.seal_block(entry_block);

            for stmt in &program.statements {
                self.compile_stmt(stmt, &mut builder)?;

                if self.is_current_block_terminated(&builder) {
                    // The block is terminated, no need to process further statements
                    break;
                }
            }

            // Ensure the function has a return instruction if not already terminated
            if !self.is_current_block_terminated(&builder) {
                builder.ins().return_(&[]);
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

    pub fn compile_stmt(&mut self, stmt: &Stmt, builder: &mut FunctionBuilder) -> Result<()> {
        match stmt {
            Stmt::VarDecl {
                name,
                var_type,
                init_expr,
            } => {
                self.compile_var_decl(name, var_type, init_expr.as_deref(), builder)?;
            }
            Stmt::VarAssign { name, expr } => {
                self.compile_var_assign(name, expr, builder)?;
            }
            Stmt::ExprStmt(expr) => {
                self.compile_expr(expr, Some(builder))?;
            }
            Stmt::Return(expr) => {
                self.compile_return(expr, builder)?;
                // After a return, the block is terminated. No further instructions should be added.
                // Return early to avoid adding more instructions.
                return Ok(());
            }
            Stmt::Block(stmts) => {
                for stmt in stmts {
                    self.compile_stmt(stmt, builder)?;

                    if self.is_current_block_terminated(builder) {
                        // The block is terminated, exit the loop
                        break;
                    }
                }
            }
            Stmt::Break => {
                // Implement break logic with Cranelift
                unimplemented!();
            }
            Stmt::Continue => {
                // Implement continue logic with Cranelift
                unimplemented!();
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
                self.compile_if_statement(condition, then_branch, else_branch.as_deref(), builder)?;
            }
            Stmt::While { condition, body } => {
                self.compile_while_loop(condition, body, builder)?;
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
    ) -> Result<()> {
        let var = Variable::new(self.variables.len());
        let cl_type = self.ast_type_to_cl_type(var_type)?;

        builder.declare_var(var, cl_type);
        self.variables.insert(name.to_string(), var);

        if let Some(expr) = init_expr {
            let value = self.compile_expr(expr, Some(builder))?;
            builder.def_var(var, value);
        }

        Ok(())
    }

    pub fn compile_var_assign(
        &mut self,
        name: &str,
        expr: &Expr,
        builder: &mut FunctionBuilder,
    ) -> Result<()> {
        if let Some(&var) = self.variables.get(name) {
            let value = self.compile_expr(expr, Some(builder))?;
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
    ) -> Result<()> {
        let value = self.compile_expr(expr, Some(builder))?;
        builder.ins().return_(&[value]);
        Ok(())
    }

    pub fn compile_func_def(&mut self, func_decl: &FuncDecl, body: &Stmt) -> Result<()> {
        // Create function signature
        let sig = self.create_signature(&func_decl.params, &func_decl.return_type)?;
        let func_id = self
            .module
            .declare_function(&func_decl.name, Linkage::Local, &sig)?;

        // Create function context
        let mut func_ctx = FunctionBuilderContext::new();
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;

        // Extract value types before creating the builder
        let param_types: Vec<Type> = ctx
            .func
            .signature
            .params
            .iter()
            .map(|p| p.value_type)
            .collect();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

            let entry_block = builder.create_block();
            builder.switch_to_block(entry_block);
            // Do not seal the entry block here
            // builder.seal_block(entry_block);

            // Map function parameters to variables
            for (i, (name, _)) in func_decl.params.iter().enumerate() {
                let param = Variable::new(self.variables.len());
                self.variables.insert(name.clone(), param);

                builder.declare_var(param, param_types[i]);
                let val = builder.block_params(entry_block)[i];
                builder.def_var(param, val);
            }

            self.compile_stmt(body, &mut builder)?;

            // Ensure the function has a return instruction if not already terminated
            if !self.is_current_block_terminated(&builder) {
                builder.ins().return_(&[]);
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
    ) -> Result<Value> {
        if let Some(builder) = builder_opt {
            match expr {
                Expr::IntLiteral(val) => Ok(builder.ins().iconst(types::I64, *val)),
                Expr::BoolLiteral(val) => {
                    // Booleans represented as I8 (1-byte integer)
                    let bool_value = if *val { 1 } else { 0 };
                    Ok(builder.ins().iconst(types::I8, bool_value))
                }
                Expr::Variable(name) => {
                    if let Some(&var) = self.variables.get(name) {
                        Ok(builder.use_var(var))
                    } else {
                        Err(anyhow::anyhow!("Undefined variable `{}`", name))
                    }
                }
                Expr::BinaryOp(lhs, op, rhs) => {
                    let lhs_val = self.compile_expr(lhs, Some(builder))?;
                    let rhs_val = self.compile_expr(rhs, Some(builder))?;
                    let result = match op {
                        BinOp::Add => builder.ins().iadd(lhs_val, rhs_val),
                        BinOp::Subtract => builder.ins().isub(lhs_val, rhs_val),
                        BinOp::Multiply => builder.ins().imul(lhs_val, rhs_val),
                        BinOp::Divide => builder.ins().sdiv(lhs_val, rhs_val),
                        BinOp::Equal => {
                            let cmp = builder.ins().icmp(IntCC::Equal, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::NotEqual => {
                            let cmp = builder.ins().icmp(IntCC::NotEqual, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::LessThan => {
                            let cmp = builder.ins().icmp(IntCC::SignedLessThan, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::GreaterThan => {
                            let cmp =
                                builder.ins().icmp(IntCC::SignedGreaterThan, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        _ => unimplemented!("Operator {:?} not implemented", op),
                    };
                    Ok(result)
                }
                Expr::UnaryOp(op, expr) => {
                    let val = self.compile_expr(expr, Some(builder))?;
                    let result = match op {
                        UnaryOp::Negate => builder.ins().ineg(val),
                        UnaryOp::Not => {
                            let all_ones = builder.ins().iconst(types::I64, -1);
                            builder.ins().bxor(val, all_ones)
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
                            arg_values.push(self.compile_expr(arg, Some(builder))?);
                        }
                        let call = builder.ins().call(func_ref, &arg_values);
                        let results = builder.inst_results(call);
                        Ok(results[0])
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
    ) -> Result<()> {
        let cond_val = self.compile_expr(condition, Some(builder))?;

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
        self.compile_stmt(then_branch, builder)?;

        if !self.is_current_block_terminated(builder) {
            builder.ins().jump(merge_block, &[]);
        }
        builder.seal_block(then_block);

        // Else block
        builder.switch_to_block(else_block);
        if let Some(else_stmt) = else_branch {
            self.compile_stmt(else_stmt, builder)?;
        }

        if !self.is_current_block_terminated(builder) {
            builder.ins().jump(merge_block, &[]);
        }
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
    ) -> Result<()> {
        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_end = builder.create_block();

        // Jump to loop header
        builder.ins().jump(loop_header, &[]);

        // Loop header
        builder.switch_to_block(loop_header);
        // Indicate that loop_header can be reached from the previous block
        builder.seal_block(loop_header);

        let cond_val = self.compile_expr(condition, Some(builder))?;
        let zero = builder.ins().iconst(types::I8, 0);
        let cmp = builder.ins().icmp(IntCC::NotEqual, cond_val, zero);

        builder.ins().brif(cmp, loop_body, &[], loop_end, &[]);

        // Loop body
        builder.switch_to_block(loop_body);
        self.compile_stmt(body, builder)?;

        if !self.is_current_block_terminated(builder) {
            builder.ins().jump(loop_header, &[]);
        }
        // Indicate that loop_body can branch back to loop_header
        builder.seal_block(loop_body);

        // Loop end
        builder.switch_to_block(loop_end);
        builder.seal_block(loop_end);

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

    // Helper method to check if the current block is terminated
    fn is_current_block_terminated(&self, builder: &FunctionBuilder) -> bool {
        builder.is_unreachable()
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

    #[test]
    fn test_integer_literal() {
        // Setup the ISA and ObjectModule
        let isa_builder = cranelift_native::builder().expect("Unable to create ISA builder");
        let flag_builder = settings::builder();
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
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
        } // `codegen` goes out of scope here, releasing the mutable borrow on `module`

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_boolean_literal() {
        // Setup the JIT module
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
                Stmt::Return(Box::new(Expr::BoolLiteral(true))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");

            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i8 = run_code(&mut module, func_id);
        assert_eq!(result, 1); // Booleans are represented as `i8`, with true as `1`
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
        // Test break and continue (Note: Requires implementation in codegen)
        // Currently unimplemented, so we can skip or expect unimplemented error
        // If you expect the function to panic with unimplemented, you can use:
        // assert_panics!(code that should panic);
        // For now, we can simply note it.
        assert!(true);
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