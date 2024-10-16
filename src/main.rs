// src/main.rs

mod ast;
mod cl_codegen;

use cranelift::codegen::settings;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use target_lexicon::Triple;

use ast::*;
use cl_codegen::CodeGenerator;


fn main() {
    // Setup the JIT module
    let builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
    let mut module = JITModule::new(builder);

    // Create a code generator (dummy usage)
    let mut codegen = CodeGenerator::new(&mut module);

    // You can perform minimal operations with `codegen` here
    // For example, compile an empty program or display a message
}
