//! On Windows: embed the icon in the .exe (taskbar, Explorer, and the shortcuts Omnistall
//! makes, which take their icon from the program).

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("icons/icon.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=could not embed the icon: {e}");
        }
    }
    println!("cargo:rerun-if-changed=icons/icon.ico");
}
