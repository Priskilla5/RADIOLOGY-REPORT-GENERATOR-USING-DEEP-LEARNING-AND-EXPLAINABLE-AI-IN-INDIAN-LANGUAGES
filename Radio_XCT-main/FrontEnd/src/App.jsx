import { useState } from 'react';
import axios from 'axios';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import './index.css';
import { 
  CalendarIcon, 
  UploadIcon, 
  UserIcon, 
  FileTextIcon, 
  SmileIcon, 
  HandHeartIcon,
  LoaderIcon,
  CheckCircleIcon,
  AlertCircleIcon
} from 'lucide-react';

function App() {
  const [form, setForm] = useState({
    first_name: '',
    last_name: '',
    dob: '',
    gender: '',
    age: '',
    organ: '',
    file: null,
  });
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState({ show: false, message: '', type: '' });

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    if (name === 'file') {
      const file = files[0];
      // File validation
      if (file && !file.type.includes('image/')) {
        showToast('Please upload a valid image file', 'error');
        return;
      }
      setForm({ ...form, file });
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      if (file) reader.readAsDataURL(file);
    } else if (name === 'dob') {
      const birthDate = new Date(value);
      const today = new Date();
      const age = today.getFullYear() - birthDate.getFullYear();
      setForm({ ...form, dob: value, age });
    } else {
      setForm({ ...form, [name]: value });
    }
  };

  const handleDateChange = (date) => {
    if (!date) return;
    
    const birthDate = new Date(date);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    if (
      today.getMonth() < birthDate.getMonth() ||
      (today.getMonth() === birthDate.getMonth() && today.getDate() < birthDate.getDate())
    ) {
      age--;
    }
    setForm({ ...form, dob: date.toISOString().split('T')[0], age });
  };

  const showToast = (message, type = 'success') => {
    setToast({ show: true, message, type });
    setTimeout(() => {
      setToast({ show: false, message: '', type: '' });
    }, 5000);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Basic form validation
    if (!form.first_name || !form.last_name || !form.dob || !form.gender || !form.organ || !form.file) {
      showToast('Please fill in all required fields', 'error');
      return;
    }
    
    setLoading(true);
    const formData = new FormData();
    const fullName = `${form.first_name} ${form.last_name}`;
    formData.append('name', fullName);
    formData.append('dob', form.dob);
    formData.append('gender', form.gender);
    formData.append('age', form.age);
    formData.append('organ', form.organ);
    formData.append('file', form.file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'Medical_Report.pdf');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      showToast('Report generated successfully!', 'success');
      
      // Reset form after successful submission
      setForm({
        first_name: '',
        last_name: '',
        dob: '',
        gender: '',
        age: '',
        organ: '',
        file: null,
      });
      setImagePreview(null);
      
    } catch (err) {
      console.error(err);
      showToast('Error generating report. Please try again.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm({
      first_name: '',
      last_name: '',
      dob: '',
      gender: '',
      age: '',
      organ: '',
      file: null,
    });
    setImagePreview(null);
    showToast('Form has been reset', 'info');
  };

  return (
    <div className="min-h-screen p-0 m-0 bg-gray-50">
      {/* Toast Notification */}
      {toast.show && (
        <div className={`fixed top-4 right-4 z-50 p-4 rounded-md shadow-lg flex items-center gap-2 max-w-md
          ${toast.type === 'success' ? 'bg-green-100 text-green-800 border-l-4 border-green-500' : 
            toast.type === 'error' ? 'bg-red-100 text-red-800 border-l-4 border-red-500' : 
            'bg-blue-100 text-blue-800 border-l-4 border-blue-500'}`}>
          {toast.type === 'success' ? <CheckCircleIcon className="w-5 h-5" /> : 
           toast.type === 'error' ? <AlertCircleIcon className="w-5 h-5" /> : 
           <FileTextIcon className="w-5 h-5" />}
          <p>{toast.message}</p>
          <button 
            className="ml-auto text-gray-500 hover:text-gray-700"
            onClick={() => setToast({ show: false, message: '', type: '' })}
          >
            ×
          </button>
        </div>
      )}

      {/* Navbar */}
      <nav className="bg-green-600 p-4 text-white shadow-md">
        <div className="container mx-auto">
          <h1 className="text-xl md:text-2xl font-semibold text-center">
            RADIOLOGY REPORT GENERATOR USING DEEP LEARNING AND EXPLAINABLE AI IN INDIAN LANGUAGES
          </h1>
        </div>
      </nav>

      {/* Content */}
      <div className="container mx-auto p-6">
        {/* Header with logo */}
        <div className="flex items-center justify-between mb-6">
          <img src="/Medical_Logo.jpeg" alt="Logo" className="w-16 md:w-24 h-auto object-contain" />
          <div className="w-24" />
        </div>

        {/* Form Section */}
        <div className="flex flex-col lg:flex-row gap-6 justify-center items-start">
          <form
            onSubmit={handleSubmit}
            className="bg-white w-full max-w-xl p-6 rounded-lg shadow-lg space-y-6 border border-green-100"
          >
            <h2 className="text-xl font-semibold text-green-700 mb-4 text-center">Patient Information</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-green-700 text-sm font-medium mb-1">
                  <UserIcon className="inline-block w-4 h-4 mr-1" /> First Name
                </label>
                <input
                  className="w-full rounded-md border border-gray-300 shadow-sm focus:ring-2 focus:ring-green-400 focus:border-green-400 px-3 py-2 text-sm"
                  name="first_name"
                  value={form.first_name}
                  required
                  onChange={handleChange}
                  placeholder="Enter first name"
                />
              </div>
              <div>
                <label className="block text-green-700 text-sm font-medium mb-1">
                  <UserIcon className="inline-block w-4 h-4 mr-1" /> Last Name
                </label>
                <input
                  className="w-full rounded-md border border-gray-300 shadow-sm focus:ring-2 focus:ring-green-400 focus:border-green-400 px-3 py-2 text-sm"
                  name="last_name"
                  value={form.last_name}
                  required
                  onChange={handleChange}
                  placeholder="Enter last name"
                />
              </div>
            </div>

            <div>
              <label className="block text-green-700 text-sm font-medium mb-1">
                <CalendarIcon className="inline-block w-4 h-4 mr-1" /> Date of Birth
              </label>
              <DatePicker
                selected={form.dob ? new Date(form.dob) : null}
                onChange={handleDateChange}
                dateFormat="yyyy-MM-dd"
                className="w-full rounded-md border border-gray-300 shadow-sm focus:ring-2 focus:ring-green-400 focus:border-green-400 px-3 py-2 text-sm"
                placeholderText="Select DOB"
                maxDate={new Date()}
                showMonthDropdown
                showYearDropdown
                yearDropdownItemNumber={100}
                scrollableYearDropdown
                dropdownMode="select"
              />
            </div>

            <div>
              <label className="block text-green-700 text-sm font-medium mb-1">
                <SmileIcon className="inline-block w-4 h-4 mr-1" /> Age
              </label>
              <input
                className="w-full rounded-md border border-gray-300 bg-gray-50 shadow-sm px-3 py-2 text-sm cursor-not-allowed"
                name="age"
                value={form.age}
                readOnly
                placeholder="Calculated from DOB"
              />
            </div>

            <div>
              <label className="block text-green-700 text-sm font-medium mb-1">
                <HandHeartIcon className="inline-block w-4 h-4 mr-1" /> Gender
              </label>
              <select
                name="gender"
                value={form.gender}
                required
                className="w-full rounded-md border border-gray-300 shadow-sm focus:ring-2 focus:ring-green-400 focus:border-green-400 px-3 py-2 text-sm"
                onChange={handleChange}
              >
                <option value="">Select Gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Others">Others</option>
              </select>
            </div>

            <div>
              <label className="block text-green-700 text-sm font-medium mb-1">
                <FileTextIcon className="inline-block w-4 h-4 mr-1" /> Organ
              </label>
              <select
                name="organ"
                value={form.organ}
                required
                onChange={handleChange}
                className="w-full rounded-md border border-gray-300 shadow-sm focus:ring-2 focus:ring-green-400 focus:border-green-400 px-3 py-2 text-sm"
              >
                <option value="">Select Organ</option>
                <option value="dental">Dental</option>
                <option value="spine">Spine</option>
                <option value="fracture">Fracture</option>
                <option value="kidney">Kidney</option>
              </select>
            </div>

            <div>
              <label className="block text-green-700 text-sm font-medium mb-1">
                <UploadIcon className="inline-block w-4 h-4 mr-1" /> Upload Scan
              </label>
              <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md hover:border-green-400 transition-colors">
                <div className="space-y-1 text-center">
                  <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <div className="flex text-sm text-gray-600">
                    <label htmlFor="file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-green-600 hover:text-green-500 focus-within:outline-none">
                      <span>Upload a file</span>
                      <input 
                        id="file-upload" 
                        name="file" 
                        type="file" 
                        accept="image/*" 
                        required 
                        onChange={handleChange} 
                        className="sr-only" 
                      />
                    </label>
                    <p className="pl-1">or drag and drop</p>
                  </div>
                  <p className="text-xs text-gray-500">PNG, JPG, GIF up to 10MB</p>
                </div>
              </div>
              {form.file && (
                <p className="mt-2 text-sm text-green-600">
                  Selected: {form.file.name}
                </p>
              )}
            </div>

            <div className="flex gap-4">
              <button
                type="submit"
                disabled={loading}
                className={`w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-md transition 
                  ${loading ? 'opacity-70 cursor-not-allowed' : ''}`}
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <LoaderIcon className="w-5 h-5 mr-2 animate-spin" />
                    Processing...
                  </span>
                ) : (
                  'Generate PDF Report'
                )}
              </button>
              
              <button
                type="button"
                onClick={handleReset}
                className="w-1/3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded-md transition"
              >
                Reset
              </button>
            </div>
          </form>

          {/* Image Preview */}
          <div className="w-full max-w-sm">
            <h3 className="text-lg font-medium mb-2 text-center text-green-700">Preview</h3>
            {imagePreview ? (
              <div className="border rounded-lg overflow-hidden shadow-lg bg-white p-2">
                <img
                  src={imagePreview}
                  alt="Uploaded Preview"
                  className="w-full h-auto object-contain rounded"
                />
                <p className="mt-2 text-sm text-center text-gray-600">
                  Image will be used for analysis
                </p>
              </div>
            ) : (
              <div className="border rounded-lg overflow-hidden shadow-lg bg-white p-8 flex flex-col items-center justify-center h-64">
                <FileTextIcon className="w-16 h-16 text-gray-300" />
                <p className="mt-4 text-gray-500">No image selected</p>
                <p className="text-sm text-gray-400">Upload an image to see preview</p>
              </div>
            )}
            
            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
              <h4 className="font-medium text-blue-700 mb-2">Information:</h4>
              <ul className="text-sm text-blue-600 space-y-1">
                <li>• Supported formats: JPEG, PNG, GIF</li>
                <li>• Maximum file size: 10MB</li>
                <li>• Higher resolution images provide better analysis</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="bg-green-700 text-white text-center py-4 mt-8">
        <p className="text-sm">© 2025 Medical Imaging System - All Rights Reserved</p>
      </footer>
    </div>
  );
}

export default App;