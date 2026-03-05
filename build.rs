fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/icons/AppIcon.ico");
        if let Err(err) = res.compile() {
            panic!("failed to compile windows resources: {err}");
        }
    }
}
