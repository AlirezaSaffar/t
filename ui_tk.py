import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torchvision.transforms as T
import threading
import os

# Set the appearance mode and color theme
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm="batch", upsample="bilinear", use_tanh=True):
        super().__init__()
        self.outermost = outermost
        use_bias = norm == "instance"
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, 4, 2, 1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        if norm == "batch":
            downnorm = nn.BatchNorm2d(inner_nc)
            upnorm = nn.BatchNorm2d(outer_nc)
        elif norm == "instance":
            downnorm = nn.InstanceNorm2d(inner_nc)
            upnorm = nn.InstanceNorm2d(outer_nc)
        else:
            raise NotImplementedError()
        if outermost:
            if upsample == "convtrans":
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1)
            else:
                upconv = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(inner_nc * 2, outer_nc, 5, padding=2))
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()] if use_tanh else [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            if upsample == "convtrans":
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, 2, 1, bias=use_bias)
            else:
                upconv = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(inner_nc, outer_nc, 5, padding=2, bias=use_bias))
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if upsample == "convtrans":
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, 2, 1, bias=use_bias)
            else:
                upconv = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(inner_nc * 2, outer_nc, 5, padding=2, bias=use_bias))
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm="batch", upsample="bilinear", use_tanh=True):
        super().__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, norm=norm, innermost=True, upsample=upsample)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm=norm, upsample=upsample)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm=norm, upsample=upsample)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm=norm, upsample=upsample)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm=norm, upsample=upsample)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm=norm, upsample=upsample, use_tanh=use_tanh)
    def forward(self, x):
        return self.model(x)

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    netG = UnetGenerator(3, 1, 8, norm='batch', upsample='bilinear', use_tanh=True)
    netG = torch.nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['model_netG_state_dict'])
    netG.eval()
    netG.to(device)
    return netG

# def preprocess_image(img, size=256):
#     transform = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
#     return transform(img).unsqueeze(0)
def preprocess_image(image_path, target_size=256):
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path.convert('RGB')

    original_size = image.size  # (width, height)

    # Resize while keeping aspect ratio, then pad
    transform = T.Compose([
        T.Resize(target_size, interpolation=Image.BICUBIC),  # Keeps aspect ratio, makes smaller side = target_size
        T.CenterCrop((target_size, target_size)),  # Crop or pad to get square
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_size


# def postprocess_output(out):
#     out = out.squeeze(0).cpu().detach()
#     out = (out + 1) / 2
#     out = torch.clamp(out, 0, 1)
#     if out.shape[0] == 1:
#         out = out.squeeze(0)
#     img = T.ToPILImage()(out)
#     return img
def postprocess_output(output_tensor, original_size):
    output = output_tensor.squeeze(0).cpu().detach()
    output = (output + 1) / 2
    output = torch.clamp(output, 0, 1)

    if output.shape[0] == 1:
        output = output.squeeze(0)
        output_image = T.ToPILImage()(output)
    else:
        output_image = T.ToPILImage()(output)

    output_image = output_image.resize(original_size, Image.BICUBIC)  # Restore to original aspect ratio
    return output_image

class ThermalMapApp:
    def __init__(self, root):
        self.root = root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_photo = None
        self.output_photo = None
        self.processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Configure window
        self.root.title("Thermal Map Generator Pro")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Create main container with padding
        main_container = ctk.CTkFrame(self.root, corner_radius=0)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header section
        self.create_header(main_container)
        
        # Control panel
        self.create_control_panel(main_container)
        
        # Status section
        self.create_status_section(main_container)
        
        # Image display section
        self.create_image_section(main_container)
        
        # Footer with device info
        self.create_footer(main_container)
        
    def create_header(self, parent):
        header_frame = ctk.CTkFrame(parent, height=80, corner_radius=15)
        header_frame.pack(fill="x", pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Title with gradient-like effect using multiple labels
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(expand=True, fill="both")
        
        main_title = ctk.CTkLabel(
            title_frame, 
            text="🔥 Thermal Map Generator Pro", 
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=("#1f538d", "#4a9eff")
        )
        main_title.pack(pady=15)
        
        subtitle = ctk.CTkLabel(
            title_frame, 
            text="AI-Powered Thermal Visualization", 
            font=ctk.CTkFont(size=14),
            text_color=("gray60", "gray40")
        )
        subtitle.pack()
        
    def create_control_panel(self, parent):
        control_frame = ctk.CTkFrame(parent, corner_radius=15)
        control_frame.pack(fill="x", pady=(0, 20))
        
        # Control buttons with icons
        button_container = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_container.pack(fill="x", padx=20, pady=20)
        
        # Model selection button
        self.btn_model = ctk.CTkButton(
            button_container,
            text="📁 Select Model (.pth)",
            command=self.choose_model,
            width=200,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
            hover_color=("#2b7a2b", "#45a045")
        )
        self.btn_model.pack(side="left", padx=(0, 15))
        
        # Image selection button
        self.btn_image = ctk.CTkButton(
            button_container,
            text="🖼️ Select Image",
            command=self.choose_image,
            width=200,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
            state="disabled"
        )
        self.btn_image.pack(side="left", padx=(0, 15))
        
        # Process button
        self.btn_process = ctk.CTkButton(
            button_container,
            text="⚡ Generate Thermal Map",
            command=self.process_image,
            width=250,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
            fg_color=("#d63031", "#e74c3c"),
            hover_color=("#b71c1c", "#c62828"),
            state="disabled"
        )
        self.btn_process.pack(side="left", padx=(0, 15))
        
        # Save button
        self.btn_save = ctk.CTkButton(
            button_container,
            text="💾 Save Result",
            command=self.save_result,
            width=150,
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=10,
            fg_color=("#6c5ce7", "#a29bfe"),
            hover_color=("#5f3dc4", "#7c4dff"),
            state="disabled"
        )
        self.btn_save.pack(side="left")
        
    def create_status_section(self, parent):
        status_frame = ctk.CTkFrame(parent, height=50, corner_radius=15)
        status_frame.pack(fill="x", pady=(0, 20))
        status_frame.pack_propagate(False)
        
        status_container = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.status_label = ctk.CTkLabel(
            status_container,
            text="🔄 Ready - Select a model to begin",
            font=ctk.CTkFont(size=13),
            anchor="w"
        )
        self.status_label.pack(side="left", fill="x", expand=True)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            status_container,
            width=200,
            height=8,
            corner_radius=4
        )
        self.progress_bar.pack(side="right", padx=(10, 0))
        self.progress_bar.set(0)
        
    def create_image_section(self, parent):
        image_frame = ctk.CTkFrame(parent, corner_radius=15)
        image_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        # Image container with labels
        image_container = ctk.CTkFrame(image_frame, fg_color="transparent")
        image_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Input image section
        input_section = ctk.CTkFrame(image_container, corner_radius=12)
        input_section.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        input_header = ctk.CTkLabel(
            input_section,
            text="📸 Input Image",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=40
        )
        input_header.pack(pady=(15, 10))
        
        self.input_frame = ctk.CTkFrame(input_section, corner_radius=8)
        self.input_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.in_label = ctk.CTkLabel(
            self.input_frame,
            text="No image selected\n\nClick 'Select Image' to load an image",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray50")
        )
        self.in_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Output image section
        output_section = ctk.CTkFrame(image_container, corner_radius=12)
        output_section.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        output_header = ctk.CTkLabel(
            output_section,
            text="🔥 Thermal Map Output",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=40
        )
        output_header.pack(pady=(15, 10))
        
        self.output_frame = ctk.CTkFrame(output_section, corner_radius=8)
        self.output_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        self.out_label = ctk.CTkLabel(
            self.output_frame,
            text="Thermal map will appear here\n\nafter processing",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray50")
        )
        self.out_label.pack(expand=True, fill="both", padx=20, pady=20)
        
    def create_footer(self, parent):
        footer_frame = ctk.CTkFrame(parent, height=40, corner_radius=15)
        footer_frame.pack(fill="x")
        footer_frame.pack_propagate(False)
        
        footer_container = ctk.CTkFrame(footer_frame, fg_color="transparent")
        footer_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        device_info = f"🖥️ Device: {self.device.type.upper()}"
        if self.device.type == 'cuda':
            device_info += f" ({torch.cuda.get_device_name()})"
        
        device_label = ctk.CTkLabel(
            footer_container,
            text=device_info,
            font=ctk.CTkFont(size=11),
            text_color=("gray60", "gray40")
        )
        device_label.pack(side="left")
        
        version_label = ctk.CTkLabel(
            footer_container,
            text="v2.0 Pro",
            font=ctk.CTkFont(size=11),
            text_color=("gray60", "gray40")
        )
        version_label.pack(side="right")
        
    def choose_model(self):
        path = filedialog.askopenfilename(
            title="Select PyTorch Model",
            filetypes=[('PyTorch Model', '*.pth'), ('All files', '*.*')]
        )
        if path:
            try:
                self.model = load_model(path, self.device)
                
                self.btn_model.configure(text="✅ Model Loaded")
                self.btn_image.configure(state="normal")
                self.update_status("✅ Model loaded successfully! Now select an image.", "green")
                
            except Exception as e:
                messagebox.showerror("Model Loading Error", f"Failed to load model:\n{str(e)}")
                self.update_status("❌ Failed to load model", "red")
    
    def choose_image(self):
        if not self.model:
            messagebox.showwarning("No Model", "Please select a model first!")
            return
            
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ('Image files', '*.png *.jpg *.jpeg *.bmp *.gif *.tiff'),
                ('PNG files', '*.png'),
                ('JPEG files', '*.jpg *.jpeg'),
                ('All files', '*.*')
            ]
        )
        if path:
            try:
                self.current_image_path = path
                input_img = Image.open(path).convert('RGB')
                self.current_input_image = input_img
                self.display_input(input_img)
                self.btn_process.configure(state="normal")
                self.update_status(f"📁 Image loaded: {os.path.basename(path)}")
                
            except Exception as e:
                messagebox.showerror("Image Loading Error", f"Failed to load image:\n{str(e)}")
                self.update_status("❌ Failed to load image", "red")
    
    def process_image(self):
        if not self.model or not hasattr(self, 'current_input_image'):
            return
            
        if self.processing:
            return
            
        # Start processing in a separate thread
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        self.processing = True
        self.root.after(0, lambda: self.btn_process.configure(state="disabled"))
        self.root.after(0, lambda: self.update_status("🔄 Processing image... Please wait"))
        
        try:
            # Update progress for preprocessing
            self.root.after(0, lambda: self.progress_bar.set(0.2))
            self.root.after(0, lambda: self.update_status("🔄 Preprocessing image..."))
            
            # Preprocess the image using your function
            tensor , orig_size = preprocess_image(self.current_input_image)
            
            # Update progress for model inference
            self.root.after(0, lambda: self.progress_bar.set(0.5))
            self.root.after(0, lambda: self.update_status("🧠 Running UNet inference..."))
            
            # Run the actual model inference
            with torch.no_grad():
                output = self.model(tensor)
            
            # Update progress for postprocessing
            self.root.after(0, lambda: self.progress_bar.set(0.8))
            self.root.after(0, lambda: self.update_status("🎨 Generating thermal visualization..."))
            
            # Postprocess the output using your function
            out_img = postprocess_output(output , orig_size)
            
            # Complete processing
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            
            self.current_output_image = out_img
            self.root.after(0, lambda: self.display_output(out_img))
            self.root.after(0, lambda: self.btn_save.configure(state="normal"))
            self.root.after(0, lambda: self.update_status("✅ Thermal map generated successfully!", "green"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", f"Failed to process image:\n{str(e)}"))
            self.root.after(0, lambda: self.update_status("❌ Processing failed", "red"))
        
        finally:
            self.processing = False
            self.root.after(0, lambda: self.btn_process.configure(state="normal"))
            self.root.after(0, lambda: self.progress_bar.set(0))
    
    def save_result(self):
        if not hasattr(self, 'current_output_image'):
            return
            
        path = filedialog.asksaveasfilename(
            title="Save Thermal Map",
            defaultextension=".png",
            filetypes=[
                ('PNG files', '*.png'),
                ('JPEG files', '*.jpg'),
                ('All files', '*.*')
            ]
        )
        if path:
            try:
                self.current_output_image.save(path)
                self.update_status(f"💾 Saved: {os.path.basename(path)}", "green")
                messagebox.showinfo("Success", f"Thermal map saved successfully!\n{path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image:\n{str(e)}")
                self.update_status("❌ Failed to save image", "red")
    
    def display_input(self, img):
        self._display_image(img, self.in_label, "Input")
        
    def display_output(self, img):
        self._display_image(img, self.out_label, "Output")
        
    def _display_image(self, img, label, image_type):
        # Get the frame size for proper scaling
        frame_width = 600  # Approximate frame width
        frame_height = 450  # Approximate frame height
        
        w, h = img.size
        scale = min(frame_width / w, frame_height / h, 1.0)  # Don't upscale
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_resized)
        
        label.configure(image=photo, text="")
        if image_type == "Input":
            self.input_photo = photo
        else:
            self.output_photo = photo
    
    def update_status(self, message, color=None):
        self.status_label.configure(text=message)
        if color == "green":
            self.status_label.configure(text_color=("#2d5a2d", "#4caf50"))
        elif color == "red":
            self.status_label.configure(text_color=("#8b2635", "#f44336"))
        else:
            self.status_label.configure(text_color=("gray10", "gray90"))

# Main application setup
def main():
    root = ctk.CTk()
    app = ThermalMapApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
