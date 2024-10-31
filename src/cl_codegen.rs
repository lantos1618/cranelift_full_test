use anyhow::Result;
use cranelift::prelude::*;
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_module::{FuncId, Linkage, Module, DataDescription};
use cranelift_jit::JITModule;
use crate::ast::*;
use std::collections::HashMap;


// Define LoopContext struct
#[derive(Debug)]
pub struct LoopContext {
    pub header_block: Block,
    pub exit_block: Block,
}

pub struct StructLayout {
    pub field_offsets: HashMap<String, usize>,
    pub total_size: usize,
}

// Add a scope management struct
#[derive(Debug)]
struct Scope {
    variables: HashMap<String, Variable>,
    variable_types: HashMap<String, AstType>,
}

impl Scope {
    fn new() -> Self {
        Self {
            variables: HashMap::new(),
            variable_types: HashMap::new(),
        }
    }
}

pub struct CodeGenerator<'a> {
    pub module: &'a mut JITModule,
    #[allow(dead_code)]
    builder_context: FunctionBuilderContext,
    struct_types: HashMap<String, Vec<(String, AstType)>>, // Struct definitions
    struct_layouts: HashMap<String, StructLayout>, // Add this field
    function_ids: HashMap<String, FuncId>, // Map function names to FuncId
    scopes: Vec<Scope>,  // Stack of scopes
    string_counter: usize,  // Add this field
}

impl<'a> CodeGenerator<'a> {
    pub fn new(module: &'a mut JITModule) -> Self {
        Self {
            module,
            builder_context: FunctionBuilderContext::new(),
            struct_types: HashMap::new(),
            struct_layouts: HashMap::new(), // Initialize the new field
            function_ids: HashMap::new(),
            scopes: vec![Scope::new()], // Initialize with global scope
            string_counter: 0,  // Initialize the counter
        }
    }

    pub fn compile_program(&mut self, program: &Program) -> Result<()> {
        // First pass: compile all struct definitions
        for stmt in &program.statements {
            if let Stmt::StructDef { name, fields } = stmt {
                self.compile_struct_def(name, fields)?;
            }
        }

        // Second pass: compile all function declarations (including external ones)
        for stmt in &program.statements {
            match stmt {
                Stmt::FuncExternDecl { name, lib: _ } => {
                    self.compile_func_extern_decl(name)?;
                }
                _ => continue,
            }
        }

        // Third pass: compile all function definitions and other statements
        let mut has_main = false;
        for stmt in &program.statements {
            match stmt {
                Stmt::StructDef { .. } | Stmt::FuncExternDecl { .. } => {
                    // Skip struct definitions and external declarations as they're already handled
                    continue;
                }
                Stmt::FuncDef { func_decl, body } => {
                    if func_decl.name == "main" {
                        has_main = true;
                    }
                    self.compile_func_def(func_decl, body)?;
                }
                _ => {
                    // If we have a non-function statement, it goes into main
                    if !has_main {
                        // Create default main function
                        let mut sig = self.module.make_signature();
                        sig.returns.push(AbiParam::new(types::I64));
                        let func_id = self.module.declare_function("main", Linkage::Export, &sig)?;

                        let mut func_ctx = FunctionBuilderContext::new();
                        let mut ctx = self.module.make_context();
                        ctx.func.signature = sig;

                        let mut loop_stack = Vec::new();

                        {
                            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
                            let entry_block = builder.create_block();
                            builder.switch_to_block(entry_block);

                            // Compile all non-function statements as part of main
                            for stmt in &program.statements {
                                match stmt {
                                    Stmt::FuncDef { .. } | Stmt::StructDef { .. } | Stmt::FuncExternDecl { .. } => {
                                        continue;
                                    }
                                    _ => {
                                        self.compile_stmt(stmt, &mut builder, &mut loop_stack)?;
                                        if self.is_current_block_terminated(&builder) {
                                            break;
                                        }
                                    }
                                }
                            }

                            if !self.is_current_block_terminated(&builder) {
                                let zero = builder.ins().iconst(types::I64, 0);
                                builder.ins().return_(&[zero]);
                            }

                            builder.seal_all_blocks();
                            builder.finalize();
                        }

                        self.module.define_function(func_id, &mut ctx)?;
                        self.function_ids.insert("main".to_string(), func_id);
                        has_main = true;
                    }
                }
            }
        }

        Ok(())
    }


    pub fn compile_stmt(
        &mut self,
        stmt: &Stmt,
        builder: &mut FunctionBuilder,
        loop_stack: &mut Vec<LoopContext>,
    ) -> Result<()> {
        match stmt {
            Stmt::VarDecl { name, var_type, init_expr } => {
                let var = Variable::new(self.current_scope().variables.len());
                let cl_type = self.ast_type_to_cl_type(var_type)?;

                builder.declare_var(var, cl_type);
                
                // Add to current scope before initializing
                self.current_scope().variables.insert(name.clone(), var);
                self.current_scope().variable_types.insert(name.clone(), var_type.clone());

                if let Some(expr) = init_expr {
                    let value = self.compile_expr(expr, Some(builder), loop_stack)?;
                    builder.def_var(var, value);
                } else {
                    let zero = builder.ins().iconst(cl_type, 0);
                    builder.def_var(var, zero);
                }
            }
            Stmt::Block(stmts) => {
                self.push_scope();
                for stmt in stmts {
                    self.compile_stmt(stmt, builder, loop_stack)?;
                    if self.is_current_block_terminated(builder) {
                        break;
                    }
                }
                self.pop_scope();
            }
            Stmt::VarAssign { name, expr } => {
                self.compile_var_assign(name, expr, builder, loop_stack)?;
            }
            Stmt::ExprStmt(expr) => {
                self.compile_expr(expr, Some(builder), loop_stack)?;
            }
            Stmt::Return(expr) => {
                self.compile_return(expr, builder, loop_stack)?;
            }
            Stmt::Break => {
                if let Some(loop_context) = loop_stack.last() {
                    builder.ins().jump(loop_context.exit_block, &[]);
                    // Since we've added a jump, the current block is terminated
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
            Stmt::FuncExternDecl { name: func_decl, lib: _ } => {
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
                    loop_stack,
                )?;
            }
            Stmt::While { condition, body } => {
                self.compile_while_loop(condition, body, builder, loop_stack)?;
            }
        }
        Ok(())
    }

    pub fn compile_var_assign(
        &mut self,
        name: &str,
        expr: &Expr,
        builder: &mut FunctionBuilder,
        loop_stack: &mut Vec<LoopContext>,
    ) -> Result<()> {
        if let Some(var) = self.get_variable(name) {
            let value = self.compile_expr(expr, Some(builder), loop_stack)?;
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
        loop_stack: &mut Vec<LoopContext>,
    ) -> Result<()> {
        let value = self.compile_expr(expr, Some(builder), loop_stack)?;
        builder.ins().return_(&[value]);
        Ok(())
    }

    pub fn compile_func_def(&mut self, func_decl: &FuncDecl, body: &Stmt) -> Result<()> {
        let sig = self.create_signature(&func_decl.params, &func_decl.return_type)?;
        let func_id = self.module.declare_function(&func_decl.name, Linkage::Local, &sig)?;

        let mut func_ctx = FunctionBuilderContext::new();
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;

        let mut loop_stack = Vec::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry_block = builder.create_block();
            builder.switch_to_block(entry_block);
            builder.append_block_params_for_function_params(entry_block);

            // Push new scope for function
            self.push_scope();

            // Create a new variable index counter for this function
            let mut var_index = 0;

            // Map function parameters to variables
            for (i, (name, param_type)) in func_decl.params.iter().enumerate() {
                let var = Variable::new(var_index);
                var_index += 1;
                
                let cl_type = self.ast_type_to_cl_type(param_type)?;
                builder.declare_var(var, cl_type);
                
                let val = builder.block_params(entry_block)[i];
                builder.def_var(var, val);
                
                self.current_scope().variables.insert(name.clone(), var);
                self.current_scope().variable_types.insert(name.clone(), param_type.clone());
            }

            // Update compile_stmt to use the var_index counter
            self.compile_stmt_with_var_index(body, &mut builder, &mut loop_stack, &mut var_index)?;

            // Pop function scope
            self.pop_scope();

            if !self.is_current_block_terminated(&builder) {
                if let AstType::Void = func_decl.return_type {
                    builder.ins().return_(&[]);
                } else {
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().return_(&[zero]);
                }
            }

            builder.seal_all_blocks();
            builder.finalize();
        }

        self.module.define_function(func_id, &mut ctx)?;
        self.function_ids.insert(func_decl.name.clone(), func_id);

        Ok(())
    }

    pub fn compile_func_extern_decl(&mut self, func_decl: &FuncDecl) -> Result<()> {
        let sig = self.create_signature(&func_decl.params, &func_decl.return_type)?;
        
        // Declare the function with Import linkage
        let func_id = self.module.declare_function(
            &func_decl.name,
            Linkage::Import,
            &sig
        )?;

        // Store the function ID for later use
        self.function_ids.insert(func_decl.name.clone(), func_id);

        Ok(())
    }

    pub fn compile_expr(
        &mut self,
        expr: &Expr,
        mut builder_opt: Option<&mut FunctionBuilder>,
        loop_stack: &mut Vec<LoopContext>,
    ) -> Result<Value> {
        if let Some(builder) = builder_opt.as_deref_mut() {
            if self.is_current_block_terminated(builder) {
                return Err(anyhow::anyhow!("Cannot add instructions to a terminated block"));
            }

            match expr {
                Expr::IntLiteral(val) => Ok(builder.ins().iconst(types::I64, *val)),
                Expr::FloatLiteral(val) => Ok(builder.ins().f64const(*val)),
                Expr::BoolLiteral(val) => {
                    let bool_value = if *val { 1 } else { 0 };
                    Ok(builder.ins().iconst(types::I64, bool_value))
                }
                Expr::Variable(name) => {
                    if let Some(var) = self.get_variable(name) {
                        Ok(builder.use_var(var))
                    } else {
                        Err(anyhow::anyhow!("Undefined variable `{}`", name))
                    }
                }
                Expr::BinaryOp(lhs, op, rhs) => {
                    let lhs_val = self.compile_expr(lhs, Some(builder), loop_stack)?;
                    let rhs_val = self.compile_expr(rhs, Some(builder), loop_stack)?;

                    let result = match op {
                        // Integer arithmetic
                        BinOp::Add => builder.ins().iadd(lhs_val, rhs_val),
                        BinOp::Subtract => builder.ins().isub(lhs_val, rhs_val),
                        BinOp::Multiply => builder.ins().imul(lhs_val, rhs_val),
                        BinOp::Divide => builder.ins().sdiv(lhs_val, rhs_val),
                        BinOp::Modulus => builder.ins().srem(lhs_val, rhs_val),

                        // Float arithmetic
                        BinOp::FAdd => builder.ins().fadd(lhs_val, rhs_val),
                        BinOp::FSub => builder.ins().fsub(lhs_val, rhs_val),
                        BinOp::FMul => builder.ins().fmul(lhs_val, rhs_val),
                        BinOp::FDiv => builder.ins().fdiv(lhs_val, rhs_val),

                        // Integer comparisons
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
                        BinOp::LessThanEqual => {
                            let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::GreaterThan => {
                            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::GreaterThanEqual => {
                            let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }

                        // Float comparisons
                        BinOp::FEqual => {
                            let cmp = builder.ins().fcmp(FloatCC::Equal, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::FNotEqual => {
                            let cmp = builder.ins().fcmp(FloatCC::NotEqual, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::FLessThan => {
                            let cmp = builder.ins().fcmp(FloatCC::LessThan, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::FLessThanEqual => {
                            let cmp = builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::FGreaterThan => {
                            let cmp = builder.ins().fcmp(FloatCC::GreaterThan, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::FGreaterThanEqual => {
                            let cmp = builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }

                        // Logical operators
                        BinOp::And => {
                            let cmp = builder.ins().band(lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }
                        BinOp::Or => {
                            let cmp = builder.ins().bor(lhs_val, rhs_val);
                            builder.ins().bmask(types::I64, cmp)
                        }

                        // Bitwise operators
                        BinOp::BitAnd => builder.ins().band(lhs_val, rhs_val),
                        BinOp::BitOr => builder.ins().bor(lhs_val, rhs_val),
                        BinOp::BitXor => builder.ins().bxor(lhs_val, rhs_val),
                        BinOp::ShiftLeft => builder.ins().ishl(lhs_val, rhs_val),
                        BinOp::ShiftRight => builder.ins().sshr(lhs_val, rhs_val),
                    };
                    Ok(result)
                }
                _ => Err(anyhow::anyhow!("Expression type not implemented")),
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
        loop_stack: &mut Vec<LoopContext>,
    ) -> Result<()> {
        let cond_val = self.compile_expr(condition, Some(builder), loop_stack)?;

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
        self.compile_stmt(then_branch, builder, loop_stack)?;

        if !self.is_current_block_terminated(builder) {
            builder.ins().jump(merge_block, &[]);
        }

        // Else block
        builder.switch_to_block(else_block);
        if let Some(else_stmt) = else_branch {
            self.compile_stmt(else_stmt, builder, loop_stack)?;
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

        let cond_val = self.compile_expr(condition, Some(builder), loop_stack)?;
        let zero = builder.ins().iconst(types::I64, 0);  // Use I64 instead of I8
        let cmp = builder.ins().icmp(IntCC::NotEqual, cond_val, zero);

        builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

        // Loop body
        builder.switch_to_block(loop_body);
        // Push new scope for loop body
        self.push_scope();
        self.compile_stmt(body, builder, loop_stack)?;
        // Pop loop body scope
        self.pop_scope();

        if !self.is_current_block_terminated(builder) {
            builder.ins().jump(loop_header, &[]);
        }

        // Seal the blocks
        builder.seal_block(loop_body);
        builder.seal_block(loop_header);

        // Pop the loop context
        loop_stack.pop();

        // Loop exit
        builder.switch_to_block(loop_exit);
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
            AstType::Float => Ok(types::F64),  // Add float type support
            AstType::Bool => Ok(types::I8),
            AstType::Void => Ok(types::INVALID),
            AstType::Struct(name) => {
                let _struct_size = self.get_struct_size(name)?;
                Ok(types::I64)
            }
            AstType::Pointer(_) => Ok(types::I64),
            AstType::Char => Ok(types::I8),
            AstType::String => Ok(types::I64),
            AstType::Array(_) => Ok(types::I64),
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

    fn compile_struct_def(&mut self, name: &str, fields: &[(String, AstType)]) -> Result<()> {
        let mut offset = 0;
        let mut field_offsets = HashMap::new();
        let mut max_alignment = 1;

        // First pass: calculate maximum alignment
        for (_, field_type) in fields {
            max_alignment = max_alignment.max(self.get_type_alignment(field_type)?);
        }

        // Second pass: calculate aligned offsets
        for (field_name, field_type) in fields {
            let alignment = self.get_type_alignment(field_type)?;
            // Align the current offset
            offset = (offset + alignment - 1) & !(alignment - 1);
            field_offsets.insert(field_name.clone(), offset);
            offset += self.get_type_size(field_type)?;
        }

        // Align the final size to the maximum alignment
        let total_size = (offset + max_alignment - 1) & !(max_alignment - 1);

        // Store the layout
        self.struct_layouts.insert(name.to_string(), StructLayout {
            field_offsets,
            total_size,
        });

        // Store the struct definition
        self.struct_types.insert(name.to_string(), fields.to_vec());
        Ok(())
    }

    fn get_struct_size(&self, struct_name: &str) -> Result<usize> {
        if let Some(layout) = self.struct_layouts.get(struct_name) {
            Ok(layout.total_size)
        } else {
            Err(anyhow::anyhow!("Undefined struct: {}", struct_name))
        }
    }

    fn get_struct_field_offset(&self, struct_name: &str, field_name: &str) -> Result<usize> {
        if let Some(layout) = self.struct_layouts.get(struct_name) {
            if let Some(&offset) = layout.field_offsets.get(field_name) {
                Ok(offset)
            } else {
                Err(anyhow::anyhow!("Field {} not found in struct {}", field_name, struct_name))
            }
        } else {
            Err(anyhow::anyhow!("Struct {} not found", struct_name))
        }
    }

    fn get_variable_type(&self, name: &str) -> Result<AstType> {
        for scope in self.scopes.iter().rev() {
            if let Some(var_type) = scope.variable_types.get(name) {
                return Ok(var_type.clone());
            }
        }
        Err(anyhow::anyhow!("Type not found for variable {}", name))
    }

    fn get_field_type(&mut self, struct_expr: &Box<Expr>, field_name: &str) -> Result<AstType> {
        match &**struct_expr {
            Expr::Variable(var_name) => {
                if let AstType::Struct(struct_name) = &self.get_variable_type(var_name)? {
                    if let Some(fields) = self.struct_types.get(struct_name) {
                        if let Some((_, field_type)) = fields.iter().find(|(name, _)| name == field_name) {
                            Ok(field_type.clone())
                        } else {
                            Err(anyhow::anyhow!("Field {} not found in struct {}", field_name, struct_name))
                        }
                    } else {
                        Err(anyhow::anyhow!("Struct {} not found", struct_name))
                    }
                } else {
                    Err(anyhow::anyhow!("Expected struct type"))
                }
            }
            Expr::StructAccess(inner_struct, inner_field) => {
                // Recursively get the type for nested access
                let inner_type = self.get_field_type(inner_struct, inner_field)?;
                if let AstType::Struct(struct_name) = inner_type {
                    if let Some(fields) = self.struct_types.get(&struct_name) {
                        if let Some((_, field_type)) = fields.iter().find(|(name, _)| name == field_name) {
                            Ok(field_type.clone())
                        } else {
                            Err(anyhow::anyhow!("Field {} not found in struct {}", field_name, struct_name))
                        }
                    } else {
                        Err(anyhow::anyhow!("Struct {} not found", struct_name))
                    }
                } else {
                    Err(anyhow::anyhow!("Expected struct type"))
                }
            }
            _ => Err(anyhow::anyhow!("Expected struct variable or field access")),
        }
    }

    fn get_expr_type(&mut self, expr: &Expr) -> Result<AstType> {
        match expr {
            Expr::IntLiteral(_) => Ok(AstType::Int),
            Expr::FloatLiteral(_) => Ok(AstType::Float),
            Expr::StringLiteral(_) => Ok(AstType::String),
            Expr::CharLiteral(_) => Ok(AstType::Char),
            Expr::Variable(name) => self.get_variable_type(name),
            Expr::StructAccess(struct_expr, field_name) => {
                self.get_field_type(struct_expr, field_name)
            }
            _ => Err(anyhow::anyhow!("Type inference not implemented for this expression")),
        }
    }

    // Add scope management methods
    fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop().expect("No scope to pop");
    }

    fn current_scope(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("No scope available")
    }

    // Update variable lookup to check all scopes from innermost to outermost
    fn get_variable(&self, name: &str) -> Option<Variable> {
        // Search through all scopes from innermost to outermost
        for scope in self.scopes.iter().rev() {
            if let Some(&var) = scope.variables.get(name) {
                return Some(var);
            }
        }
        None
    }

    // Add this helper function to get type alignment
    fn get_type_alignment(&self, ast_type: &AstType) -> Result<usize> {
        match ast_type {
            AstType::Int => Ok(8),  // 64-bit alignment
            AstType::Bool => Ok(1), // 8-bit alignment
            AstType::Char => Ok(1), // 8-bit alignment
            AstType::String => Ok(8), // Pointer alignment
            AstType::Pointer(_) => Ok(8), // Pointer alignment
            AstType::Struct(name) => {
                // For structs, use the maximum alignment of their fields
                if let Some(fields) = self.struct_types.get(name) {
                    let mut max_alignment = 1;
                    for (_, field_type) in fields {
                        max_alignment = max_alignment.max(self.get_type_alignment(field_type)?);
                    }
                    Ok(max_alignment)
                } else {
                    Err(anyhow::anyhow!("Undefined struct: {}", name))
                }
            }
            AstType::Array(elem_type) => self.get_type_alignment(elem_type),
            AstType::Void => Ok(1),
            _ => Err(anyhow::anyhow!("Unsupported type for alignment calculation: {:?}", ast_type))
        }
    }

    fn get_type_size(&self, ast_type: &AstType) -> Result<usize> {
        match ast_type {
            AstType::Int => Ok(8),
            AstType::Bool => Ok(1),
            AstType::Char => Ok(1),
            AstType::String => Ok(16), // pointer (8) + length (8)
            AstType::Pointer(_) => Ok(8),
            AstType::Array(elem_type) => {
                let elem_size = self.get_type_size(elem_type)?;
                Ok(8 * elem_size) // Assuming fixed size arrays of 8 elements
            }
            AstType::Struct(name) => self.get_struct_size(name),
            AstType::Void => Ok(0),
            _ => Err(anyhow::anyhow!("Unsupported type for size calculation: {:?}", ast_type))
        }
    }

    fn compile_string_literal(&mut self, s: &str, builder: &mut FunctionBuilder) -> Result<Value> {
        let mut data = DataDescription::new();
        
        // Add null terminator for C strings
        let mut string_data = s.as_bytes().to_vec();
        string_data.push(0);
        
        data.define(string_data.into_boxed_slice());
        
        // Create a unique name for the string
        let name = format!("str_{}", self.next_string_id());
        
        // Declare the data in the module
        let data_id = self.module.declare_data(
            &name,
            Linkage::Local,
            true,  // readonly
            false, // not thread local
        )?;
        
        // Define the data
        self.module.define_data(data_id, &data)?;
        
        // Get a reference to the data in the function
        let local_id = self.module.declare_data_in_func(data_id, builder.func);
        
        // Get the address of the string
        Ok(builder.ins().symbol_value(types::I64, local_id))
    }

    fn next_string_id(&mut self) -> usize {
        let id = self.string_counter;
        self.string_counter += 1;
        id
    }

    // Add this method
    fn pointer_type(&self) -> Type {
        types::I64 // Use 64-bit pointers
    }

    // Add a new method to handle variable indexing
    fn compile_stmt_with_var_index(
        &mut self,
        stmt: &Stmt,
        builder: &mut FunctionBuilder,
        loop_stack: &mut Vec<LoopContext>,
        var_index: &mut usize,
    ) -> Result<()> {
        match stmt {
            Stmt::VarDecl { name, var_type, init_expr } => {
                let var = Variable::new(*var_index);
                *var_index += 1;
                
                let cl_type = self.ast_type_to_cl_type(var_type)?;
                builder.declare_var(var, cl_type);
                
                // Add to current scope before initializing
                self.current_scope().variables.insert(name.clone(), var);
                self.current_scope().variable_types.insert(name.clone(), var_type.clone());

                if let Some(expr) = init_expr {
                    let value = self.compile_expr(expr, Some(builder), loop_stack)?;
                    builder.def_var(var, value);
                } else {
                    let zero = builder.ins().iconst(cl_type, 0);
                    builder.def_var(var, zero);
                }
                Ok(())
            }
            Stmt::Block(stmts) => {
                self.push_scope();
                for stmt in stmts {
                    self.compile_stmt_with_var_index(stmt, builder, loop_stack, var_index)?;
                    if self.is_current_block_terminated(builder) {
                        break;
                    }
                }
                self.pop_scope();
                Ok(())
            }
            // Handle other statement types by delegating to the original compile_stmt
            _ => self.compile_stmt(stmt, builder, loop_stack),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_jit::JITBuilder;
    use std::mem::transmute;

    // Helper function for tests
    fn run_code<T>(module: &mut JITModule, func_id: FuncId) -> T {
        // Finalize the function definitions
        module.finalize_definitions().unwrap();

        // Get a pointer to the function's code
        let code = module.get_finalized_function(func_id);

        // Cast the code pointer to a callable function
        unsafe { transmute::<_, fn() -> T>(code)() }
    }

    fn setup_test_codegen() -> CodeGenerator<'static> {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = Box::leak(Box::new(JITModule::new(builder)));
        CodeGenerator::new(module)
    }

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
        let program = Program::new(vec![
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
                        Box::new(Expr::Variable("y".to_string()))
                    ))),
                ])),
            },
            Stmt::FuncDef {
                func_decl: FuncDecl {
                    name: "main".to_string(),
                    params: vec![],
                    return_type: AstType::Int,
                },
                body: Box::new(Stmt::Block(vec![
                    Stmt::Return(Box::new(Expr::FuncCall(
                        "add".to_string(),
                        vec![
                            Expr::IntLiteral(1),
                            Expr::IntLiteral(2)
                        ]
                    )))
                ])),
            }
        ]);

        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let mut codegen = CodeGenerator::new(&mut module);
        let result = codegen.compile_program(&program);

        assert!(result.is_ok());
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
        let program = Program::new(vec![
            Stmt::FuncDef {
                func_decl: FuncDecl {
                    name: "main".to_string(),
                    params: vec![],
                    return_type: AstType::Int,
                },
                body: Box::new(Stmt::Block(vec![
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
                ])),
            },
        ]);
        
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let mut codegen = CodeGenerator::new(&mut module);
        let result = codegen.compile_program(&program);

        assert!(result.is_ok());
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

    #[test]
    fn test_struct_definition() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let mut codegen = CodeGenerator::new(&mut module);

        // Define a struct
        let fields = vec![
            ("x".to_string(), AstType::Int),      // 8 bytes, offset 0
            ("y".to_string(), AstType::Int),      // 8 bytes, offset 8
            ("valid".to_string(), AstType::Bool), // 1 byte, offset 16, padded to 24
        ];

        assert!(codegen.compile_struct_def("Point", &fields).is_ok());

        // Test struct size calculation (24 bytes due to alignment)
        assert_eq!(codegen.get_struct_size("Point").unwrap(), 24);

        // Test field offset calculation
        assert_eq!(codegen.get_struct_field_offset("Point", "x").unwrap(), 0);
        assert_eq!(codegen.get_struct_field_offset("Point", "y").unwrap(), 8);
        assert_eq!(codegen.get_struct_field_offset("Point", "valid").unwrap(), 16);
    }

    #[test]
    fn test_struct_access_and_assignment() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        {
            let mut codegen = CodeGenerator::new(&mut module);

            // Define the Point struct
            let fields = vec![
                ("x".to_string(), AstType::Int),
                ("y".to_string(), AstType::Int),
            ];
            codegen.compile_struct_def("Point", &fields).unwrap();

            // Test struct size calculation (16 bytes: 8 for x + 8 for y)
            assert_eq!(codegen.get_struct_size("Point").unwrap(), 16);

            // Test field offset calculation
            assert_eq!(codegen.get_struct_field_offset("Point", "x").unwrap(), 0);
            assert_eq!(codegen.get_struct_field_offset("Point", "y").unwrap(), 8);
            
            // Remove the invalid test for "valid" field
            // assert_eq!(codegen.get_struct_field_offset("Point", "valid").unwrap(), 16);
        }

        // Test struct access and assignment
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        {
            let mut codegen = CodeGenerator::new(&mut module);

            // Define the Point struct
            let fields = vec![
                ("x".to_string(), AstType::Int),
                ("y".to_string(), AstType::Int),
            ];
            codegen.compile_struct_def("Point", &fields).unwrap();

            // Test struct size calculation (16 bytes: 8 for x + 8 for y)
            assert_eq!(codegen.get_struct_size("Point").unwrap(), 16);

            // Test field offset calculation
            assert_eq!(codegen.get_struct_field_offset("Point", "x").unwrap(), 0);
            assert_eq!(codegen.get_struct_field_offset("Point", "y").unwrap(), 8);
            
            // Remove the invalid test for "valid" field
            // assert_eq!(codegen.get_struct_field_offset("Point", "valid").unwrap(), 16);
        }
    }

    #[test]
    fn test_type_sizes() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let codegen = CodeGenerator::new(&mut module);

        assert_eq!(codegen.get_type_size(&AstType::Int).unwrap(), 8);
        assert_eq!(codegen.get_type_size(&AstType::Bool).unwrap(), 1);
        assert_eq!(codegen.get_type_size(&AstType::Char).unwrap(), 1);
        assert_eq!(codegen.get_type_size(&AstType::String).unwrap(), 16);
        assert_eq!(codegen.get_type_size(&AstType::Pointer(Box::new(AstType::Int))).unwrap(), 8);
    }

    #[test]
    fn test_nested_struct() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            // Define the inner Point struct
            let point_fields = vec![
                ("x".to_string(), AstType::Int),
                ("y".to_string(), AstType::Int),
            ];
            codegen.compile_struct_def("Point", &point_fields).unwrap();

            // Define the Rectangle struct that contains two Points
            let rect_fields = vec![
                ("top_left".to_string(), AstType::Struct("Point".to_string())),
                ("bottom_right".to_string(), AstType::Struct("Point".to_string())),
            ];
            codegen.compile_struct_def("Rectangle", &rect_fields).unwrap();

            // Create a program that uses nested structs
            let prog = Program::new(vec![
                // Create a rectangle
                Stmt::VarDecl {
                    name: "rect".to_string(),
                    var_type: AstType::Struct("Rectangle".to_string()),
                    init_expr: Some(Box::new(Expr::StructInit {
                        struct_name: "Rectangle".to_string(),
                        fields: vec![
                            ("top_left".to_string(), Expr::StructInit {
                                struct_name: "Point".to_string(),
                                fields: vec![
                                    ("x".to_string(), Expr::IntLiteral(0)),
                                    ("y".to_string(), Expr::IntLiteral(10)),
                                ],
                            }),
                            ("bottom_right".to_string(), Expr::StructInit {
                                struct_name: "Point".to_string(),
                                fields: vec![
                                    ("x".to_string(), Expr::IntLiteral(20)),
                                    ("y".to_string(), Expr::IntLiteral(0)),
                                ],
                            }),
                        ],
                    })),
                },
                // Calculate and return width * height
                Stmt::Return(Box::new(Expr::BinaryOp(
                    Box::new(Expr::BinaryOp(
                        Box::new(Expr::StructAccess(
                            Box::new(Expr::StructAccess(
                                Box::new(Expr::Variable("rect".to_string())),
                                "bottom_right".to_string(),
                            )),
                            "x".to_string(),
                        )),
                        BinOp::Subtract,
                        Box::new(Expr::StructAccess(
                            Box::new(Expr::StructAccess(
                                Box::new(Expr::Variable("rect".to_string())),
                                "top_left".to_string(),
                            )),
                            "x".to_string(),
                        )),
                    )),
                    BinOp::Multiply,
                    Box::new(Expr::BinaryOp(
                        Box::new(Expr::StructAccess(
                            Box::new(Expr::StructAccess(
                                Box::new(Expr::Variable("rect".to_string())),
                                "top_left".to_string(),
                            )),
                            "y".to_string(),
                        )),
                        BinOp::Subtract,
                        Box::new(Expr::StructAccess(
                            Box::new(Expr::StructAccess(
                                Box::new(Expr::Variable("rect".to_string())),
                                "bottom_right".to_string(),
                            )),
                            "y".to_string(),
                        )),
                    )),
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");
            func_id = codegen.get_function_id("main").unwrap();
        }

        // Execute the compiled code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 200); // width (20-0) * height (10-0) = 20 * 10 = 200
    }

    #[test]
    fn test_struct_array() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let mut codegen = CodeGenerator::new(&mut module);

        // Define a struct with an array field
        let fields = vec![
            ("data".to_string(), AstType::Array(Box::new(AstType::Int))),
            ("length".to_string(), AstType::Int),
        ];

        assert!(codegen.compile_struct_def("IntArray", &fields).is_ok());

        // Test struct size calculation (8 * 8 for array + 8 for length)
        assert_eq!(codegen.get_struct_size("IntArray").unwrap(), 72);
    }

    #[test]
    fn test_complex_struct() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let mut codegen = CodeGenerator::new(&mut module);

        // Define a complex struct with multiple types
        let fields = vec![
            ("name".to_string(), AstType::String),       // offset 0  (8-byte aligned)
            ("age".to_string(), AstType::Int),           // offset 16 (8-byte aligned)
            ("is_active".to_string(), AstType::Bool),    // offset 24 (1-byte aligned)
            ("points".to_string(), AstType::Array(Box::new(AstType::Int))), // offset 32 (8-byte aligned)
            ("next".to_string(), AstType::Pointer(Box::new(AstType::Struct("Person".to_string())))), // offset 96 (8-byte aligned)
        ];

        assert!(codegen.compile_struct_def("Person", &fields).is_ok());

        // Test struct size calculation
        // String (16) + padding (0) +
        // Int (8) + padding (0) +
        // Bool (1) + padding (7) +
        // Array (64) + padding (0) +
        // Pointer (8) = 104
        assert_eq!(codegen.get_struct_size("Person").unwrap(), 104);

        // Test field offsets
        assert_eq!(codegen.get_struct_field_offset("Person", "name").unwrap(), 0);
        assert_eq!(codegen.get_struct_field_offset("Person", "age").unwrap(), 16);
        assert_eq!(codegen.get_struct_field_offset("Person", "is_active").unwrap(), 24);
        assert_eq!(codegen.get_struct_field_offset("Person", "points").unwrap(), 32);
        assert_eq!(codegen.get_struct_field_offset("Person", "next").unwrap(), 96);
    }

    #[test]
    fn test_function_variable_scope() {
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
                Stmt::FuncDef {
                    func_decl: FuncDecl {
                        name: "test_scope".to_string(),
                        params: vec![],
                        return_type: AstType::Int,
                    },
                    body: Box::new(Stmt::Block(vec![
                        Stmt::VarDecl {
                            name: "x".to_string(),
                            var_type: AstType::Int,
                            init_expr: Some(Box::new(Expr::IntLiteral(42))),
                        },
                        Stmt::Return(Box::new(Expr::Variable("x".to_string()))),
                    ])),
                },
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");
            func_id = codegen.get_function_id("test_scope").unwrap();
        }

        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_extern_function() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let mut codegen = CodeGenerator::new(&mut module);

        // Declare an external function (e.g., puts from libc)
        let func_decl = FuncDecl {
            name: "puts".to_string(),
            params: vec![("s".to_string(), AstType::Pointer(Box::new(AstType::Char)))],
            return_type: AstType::Int,
        };

        // Compile the external declaration
        assert!(codegen.compile_func_extern_decl(&func_decl).is_ok());

        // Verify that the function ID was stored
        assert!(codegen.get_function_id("puts").is_some());
    }

    #[test]
    fn test_extern_function_call() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let mut codegen = CodeGenerator::new(&mut module);

        // Create a program that declares and uses puts
        let program = Program::new(vec![
            // Declare the external function
            Stmt::FuncExternDecl {
                name: FuncDecl {
                    name: "puts".to_string(),
                    params: vec![("s".to_string(), AstType::Pointer(Box::new(AstType::Char)))],
                    return_type: AstType::Int,
                },
                lib: "libc".to_string(),
            },
            // Define main function that calls puts
            Stmt::FuncDef {
                func_decl: FuncDecl {
                    name: "main".to_string(),
                    params: vec![],
                    return_type: AstType::Int,
                },
                body: Box::new(Stmt::Block(vec![
                    // Call puts with a string literal
                    Stmt::ExprStmt(Box::new(Expr::FuncCall(
                        "puts".to_string(),
                        vec![Expr::StringLiteral("Hello from external function!".to_string())],
                    ))),
                    // Return 0
                    Stmt::Return(Box::new(Expr::IntLiteral(0))),
                ])),
            },
        ]);

        // Compile the program and print any error
        match codegen.compile_program(&program) {
            Ok(_) => (),
            Err(e) => panic!("Failed to compile program: {}", e),
        }

        // Get the main function ID
        let func_id = codegen.get_function_id("main").unwrap();

        // Run the code
        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_float_arithmetic() {
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
                    Box::new(Expr::FloatLiteral(3.14)),
                    BinOp::FAdd,
                    Box::new(Expr::FloatLiteral(2.86)),
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");
            func_id = codegen.get_function_id("main").unwrap();
        }

        let result: f64 = unsafe { std::mem::transmute(run_code::<i64>(&mut module, func_id)) };
        assert!((result - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_float_comparison() {
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
                    Box::new(Expr::FloatLiteral(3.14)),
                    BinOp::FLessThan,
                    Box::new(Expr::FloatLiteral(3.15)),
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");
            func_id = codegen.get_function_id("main").unwrap();
        }

        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 1); // true because 3.14 < 3.15
    }

    #[test]
    fn test_bitwise_operations() {
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
                    Box::new(Expr::IntLiteral(0b1100)),
                    BinOp::BitAnd,
                    Box::new(Expr::IntLiteral(0b1010)),
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");
            func_id = codegen.get_function_id("main").unwrap();
        }

        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 0b1000); // 12 & 10 = 8
    }

    #[test]
    fn test_logical_operations() {
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
                    Box::new(Expr::BoolLiteral(true)),
                    BinOp::And,
                    Box::new(Expr::BoolLiteral(false)),
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");
            func_id = codegen.get_function_id("main").unwrap();
        }

        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 0); // true AND false = false
    }

    #[test]
    fn test_shift_operations() {
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
                    Box::new(Expr::IntLiteral(8)),
                    BinOp::ShiftLeft,
                    Box::new(Expr::IntLiteral(2)),
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");
            func_id = codegen.get_function_id("main").unwrap();
        }

        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 32); // 8 << 2 = 32
    }

    #[test]
    fn test_mixed_float_int_comparison() {
        let isa = cranelift_native::builder()
            .unwrap()
            .finish(settings::Flags::new(settings::builder()))
            .unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        let func_id;

        {
            let mut codegen = CodeGenerator::new(&mut module);

            // Test a more complex expression with both float and int operations
            let prog = Program::new(vec![
                Stmt::VarDecl {
                    name: "x".to_string(),
                    var_type: AstType::Float,
                    init_expr: Some(Box::new(Expr::FloatLiteral(3.14))),
                },
                Stmt::VarDecl {
                    name: "y".to_string(),
                    var_type: AstType::Float,
                    init_expr: Some(Box::new(Expr::FloatLiteral(2.0))),
                },
                Stmt::Return(Box::new(Expr::BinaryOp(
                    Box::new(Expr::BinaryOp(
                        Box::new(Expr::Variable("x".to_string())),
                        BinOp::FMul,
                        Box::new(Expr::Variable("y".to_string())),
                    )),
                    BinOp::FGreaterThan,
                    Box::new(Expr::FloatLiteral(6.0)),
                ))),
            ]);

            codegen.compile_program(&prog).expect("Compilation failed");
            func_id = codegen.get_function_id("main").unwrap();
        }

        let result: i64 = run_code(&mut module, func_id);
        assert_eq!(result, 1); // 3.14 * 2.0 > 6.0 is true
    }
}
























