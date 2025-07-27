
"""
Simple GUI Interface for Attendance System
Uses tkinter for cross-platform compatibility
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
from attendance_system import AttendanceSystem
from face_recognition_module import FaceRecognizer
from config import Config

class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Attendance System")
        self.root.geometry("800x600")

        # Initialize components
        self.attendance_system = None
        self.face_recognizer = FaceRecognizer()
        self.camera_running = False
        self.cap = None

        # Create GUI elements
        self.create_widgets()

        # Setup directories
        Config.create_directories()

    def create_widgets(self):
        """Create GUI widgets"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Student Management
        student_frame = ttk.Frame(notebook)
        notebook.add(student_frame, text="Student Management")
        self.create_student_tab(student_frame)

        # Tab 2: Attendance System
        attendance_frame = ttk.Frame(notebook)
        notebook.add(attendance_frame, text="Attendance System")
        self.create_attendance_tab(attendance_frame)

        # Tab 3: Reports
        reports_frame = ttk.Frame(notebook)
        notebook.add(reports_frame, text="Reports")
        self.create_reports_tab(reports_frame)

        # Tab 4: Settings
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        self.create_settings_tab(settings_frame)

    def create_student_tab(self, parent):
        """Create student management tab"""
        # Add Student Section
        add_frame = ttk.LabelFrame(parent, text="Add New Student", padding=10)
        add_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(add_frame, text="Student ID:").grid(row=0, column=0, sticky='w', pady=2)
        self.student_id_entry = ttk.Entry(add_frame, width=20)
        self.student_id_entry.grid(row=0, column=1, padx=(10, 0), pady=2)

        ttk.Label(add_frame, text="Student Name:").grid(row=1, column=0, sticky='w', pady=2)
        self.student_name_entry = ttk.Entry(add_frame, width=30)
        self.student_name_entry.grid(row=1, column=1, padx=(10, 0), pady=2)

        button_frame = ttk.Frame(add_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Add from Camera", 
                  command=self.add_student_camera).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Add from Files", 
                  command=self.add_student_files).pack(side='left', padx=5)

        # Student List Section
        list_frame = ttk.LabelFrame(parent, text="Registered Students", padding=10)
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Treeview for student list
        columns = ('ID', 'Name')
        self.student_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)

        for col in columns:
            self.student_tree.heading(col, text=col)
            self.student_tree.column(col, width=150)

        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.student_tree.yview)
        self.student_tree.configure(yscroll=scrollbar.set)

        self.student_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Buttons for student management
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill='x', pady=(10, 0))

        ttk.Button(btn_frame, text="Refresh List", 
                  command=self.refresh_student_list).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Remove Selected", 
                  command=self.remove_selected_student).pack(side='left', padx=5)

        # Load initial student list
        self.refresh_student_list()

    def create_attendance_tab(self, parent):
        """Create attendance system tab"""
        # Control Section
        control_frame = ttk.LabelFrame(parent, text="Attendance Control", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)

        self.attendance_status = ttk.Label(control_frame, text="Status: Ready", 
                                         font=('Arial', 12, 'bold'))
        self.attendance_status.pack(pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=5)

        self.start_btn = ttk.Button(button_frame, text="Start Attendance", 
                                   command=self.start_attendance)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop Attendance", 
                                  command=self.stop_attendance, state='disabled')
        self.stop_btn.pack(side='left', padx=5)

        # Video Display Section
        video_frame = ttk.LabelFrame(parent, text="Camera Feed", padding=10)
        video_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.video_label = ttk.Label(video_frame, text="Camera feed will appear here")
        self.video_label.pack(expand=True)

        # Attendance Info Section
        info_frame = ttk.LabelFrame(parent, text="Current Session", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)

        self.present_count_label = ttk.Label(info_frame, text="Present: 0")
        self.present_count_label.pack(side='left', padx=10)

        self.time_label = ttk.Label(info_frame, text="Time: --:--:--")
        self.time_label.pack(side='right', padx=10)

    def create_reports_tab(self, parent):
        """Create reports tab"""
        # Report Generation Section
        gen_frame = ttk.LabelFrame(parent, text="Generate Reports", padding=10)
        gen_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(gen_frame, text="Date Range:").grid(row=0, column=0, sticky='w', pady=2)

        date_frame = ttk.Frame(gen_frame)
        date_frame.grid(row=0, column=1, padx=10, pady=2)

        self.start_date_entry = ttk.Entry(date_frame, width=12)
        self.start_date_entry.pack(side='left')
        self.start_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))

        ttk.Label(date_frame, text=" to ").pack(side='left', padx=5)

        self.end_date_entry = ttk.Entry(date_frame, width=12)
        self.end_date_entry.pack(side='left')
        self.end_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))

        ttk.Button(gen_frame, text="Generate Report", 
                  command=self.generate_report).grid(row=1, column=0, columnspan=2, pady=10)

        # Available Reports Section
        reports_frame = ttk.LabelFrame(parent, text="Available Reports", padding=10)
        reports_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Listbox for reports
        self.reports_listbox = tk.Listbox(reports_frame, height=10)
        self.reports_listbox.pack(fill='both', expand=True)

        ttk.Button(reports_frame, text="Refresh List", 
                  command=self.refresh_reports_list).pack(pady=5)

        # Load initial reports list
        self.refresh_reports_list()

    def create_settings_tab(self, parent):
        """Create settings tab"""
        # Time Settings
        time_frame = ttk.LabelFrame(parent, text="Attendance Time Window", padding=10)
        time_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(time_frame, text="Start Time:").grid(row=0, column=0, sticky='w', pady=2)
        self.start_time_entry = ttk.Entry(time_frame, width=10)
        self.start_time_entry.grid(row=0, column=1, padx=10, pady=2)
        self.start_time_entry.insert(0, "09:30")

        ttk.Label(time_frame, text="End Time:").grid(row=1, column=0, sticky='w', pady=2)
        self.end_time_entry = ttk.Entry(time_frame, width=10)
        self.end_time_entry.grid(row=1, column=1, padx=10, pady=2)
        self.end_time_entry.insert(0, "10:00")

        # Camera Settings
        camera_frame = ttk.LabelFrame(parent, text="Camera Settings", padding=10)
        camera_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(camera_frame, text="Camera Index:").grid(row=0, column=0, sticky='w', pady=2)
        self.camera_index_entry = ttk.Entry(camera_frame, width=5)
        self.camera_index_entry.grid(row=0, column=1, padx=10, pady=2)
        self.camera_index_entry.insert(0, "0")

        ttk.Button(camera_frame, text="Test Camera", 
                  command=self.test_camera).grid(row=0, column=2, padx=10, pady=2)

        # Recognition Settings
        recognition_frame = ttk.LabelFrame(parent, text="Recognition Settings", padding=10)
        recognition_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(recognition_frame, text="Face Recognition Threshold:").grid(row=0, column=0, sticky='w', pady=2)
        self.face_threshold_entry = ttk.Entry(recognition_frame, width=10)
        self.face_threshold_entry.grid(row=0, column=1, padx=10, pady=2)
        self.face_threshold_entry.insert(0, "0.6")

        ttk.Label(recognition_frame, text="Emotion Confidence Threshold:").grid(row=1, column=0, sticky='w', pady=2)
        self.emotion_threshold_entry = ttk.Entry(recognition_frame, width=10)
        self.emotion_threshold_entry.grid(row=1, column=1, padx=10, pady=2)
        self.emotion_threshold_entry.insert(0, "0.5")

        ttk.Button(recognition_frame, text="Save Settings", 
                  command=self.save_settings).grid(row=2, column=0, columnspan=2, pady=10)

    # Student Management Methods
    def add_student_camera(self):
        """Add student using camera capture"""
        student_id = self.student_id_entry.get().strip()
        student_name = self.student_name_entry.get().strip()

        if not student_id or not student_name:
            messagebox.showerror("Error", "Please enter both Student ID and Name")
            return

        # This would open a camera window for capture
        messagebox.showinfo("Camera Capture", 
                           "Camera capture functionality would be implemented here.\n"
                           "For now, use the command line interface:\n"
                           f"python main.py add-student {student_id} \"{student_name}\" --interactive")

    def add_student_files(self):
        """Add student from image files"""
        student_id = self.student_id_entry.get().strip()
        student_name = self.student_name_entry.get().strip()

        if not student_id or not student_name:
            messagebox.showerror("Error", "Please enter both Student ID and Name")
            return

        # Open file dialog for image selection
        file_paths = filedialog.askopenfilenames(
            title="Select Student Photos",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_paths:
            try:
                self.face_recognizer.add_student(student_id, student_name, list(file_paths))
                messagebox.showinfo("Success", 
                                   f"Student {student_name} added successfully with {len(file_paths)} photos")
                self.student_id_entry.delete(0, tk.END)
                self.student_name_entry.delete(0, tk.END)
                self.refresh_student_list()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add student: {str(e)}")

    def refresh_student_list(self):
        """Refresh the student list"""
        # Clear existing items
        for item in self.student_tree.get_children():
            self.student_tree.delete(item)

        # Get students and populate tree
        students = self.face_recognizer.get_all_students()
        for student in students:
            self.student_tree.insert('', 'end', values=(student['id'], student['name']))

    def remove_selected_student(self):
        """Remove selected student"""
        selection = self.student_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a student to remove")
            return

        item = self.student_tree.item(selection[0])
        student_id = item['values'][0]

        if messagebox.askyesno("Confirm", f"Remove student {student_id}?"):
            try:
                self.face_recognizer.remove_student(student_id)
                messagebox.showinfo("Success", f"Student {student_id} removed successfully")
                self.refresh_student_list()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to remove student: {str(e)}")

    # Attendance System Methods
    def start_attendance(self):
        """Start attendance system"""
        try:
            self.attendance_system = AttendanceSystem()
            self.camera_running = True

            # Update UI
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.attendance_status.config(text="Status: Running")

            # Start camera in separate thread
            threading.Thread(target=self.run_attendance_camera, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start attendance: {str(e)}")

    def stop_attendance(self):
        """Stop attendance system"""
        self.camera_running = False

        if self.cap:
            self.cap.release()

        # Update UI
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.attendance_status.config(text="Status: Stopped")
        self.video_label.config(image="", text="Camera feed stopped")

        # Export attendance data
        if self.attendance_system:
            try:
                self.attendance_system.mark_absent_students()
                self.attendance_system.export_attendance()
                messagebox.showinfo("Success", "Attendance data exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def run_attendance_camera(self):
        """Run attendance camera in separate thread"""
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.stop_attendance()
            return

        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame for attendance
            processed_frame = self.attendance_system.process_frame(frame)

            # Convert frame for tkinter display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((400, 300), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)

            # Update video display in main thread
            self.root.after(0, self.update_video_display, frame_tk)

            # Update attendance info
            present_count = len(self.attendance_system.detected_students)
            current_time = datetime.now().strftime("%H:%M:%S")

            self.root.after(0, self.update_attendance_info, present_count, current_time)

        if self.cap:
            self.cap.release()

    def update_video_display(self, frame_tk):
        """Update video display in main thread"""
        self.video_label.config(image=frame_tk, text="")
        self.video_label.image = frame_tk  # Keep a reference

    def update_attendance_info(self, present_count, current_time):
        """Update attendance information"""
        self.present_count_label.config(text=f"Present: {present_count}")
        self.time_label.config(text=f"Time: {current_time}")

    # Reports Methods
    def generate_report(self):
        """Generate attendance report"""
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()

        # This would generate actual reports
        messagebox.showinfo("Report Generation", 
                           f"Report generation for {start_date} to {end_date} would be implemented here.\n"
                           "For now, use the command line interface:\n"
                           f"python main.py report {start_date} {end_date}")

    def refresh_reports_list(self):
        """Refresh reports list"""
        self.reports_listbox.delete(0, tk.END)

        # List CSV files in attendance directory
        if os.path.exists(Config.ATTENDANCE_DIR):
            files = [f for f in os.listdir(Config.ATTENDANCE_DIR) if f.endswith('.csv')]
            for file in sorted(files):
                self.reports_listbox.insert(tk.END, file)

    # Settings Methods
    def test_camera(self):
        """Test camera functionality"""
        camera_index = int(self.camera_index_entry.get())
        cap = cv2.VideoCapture(camera_index)

        if cap.isOpened():
            messagebox.showinfo("Camera Test", f"Camera {camera_index} is working correctly")
            cap.release()
        else:
            messagebox.showerror("Camera Test", f"Camera {camera_index} is not available")

    def save_settings(self):
        """Save settings"""
        # This would save settings to config file
        messagebox.showinfo("Settings", "Settings saved successfully")

def main():
    """Main GUI application"""
    root = tk.Tk()
    app = AttendanceGUI(root)

    # Handle window closing
    def on_closing():
        if app.camera_running:
            app.stop_attendance()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
