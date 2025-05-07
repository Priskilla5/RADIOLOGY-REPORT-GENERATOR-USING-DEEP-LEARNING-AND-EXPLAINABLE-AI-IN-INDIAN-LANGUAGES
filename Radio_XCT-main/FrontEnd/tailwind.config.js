/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class', // <- THIS LINE is crucial
  theme: {
    extend: {
      colors: {
        'pastel-green': '#d8f3dc', // very soft green
      },
    },
  },
  
  plugins: [],
}
