// src/main.rs

mod ast;
mod cl_codegen;
mod cl_codegen_test;

use cranelift::codegen::settings;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;
use target_lexicon::Triple;

use ast::*;
use cl_codegen::CodeGen;

fn main() {
  
}
