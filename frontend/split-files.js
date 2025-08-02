const fs = require('fs');
const path = require('path');

// Read your pasted content
const content = fs.readFileSync('all_files_combined.txt', 'utf8');

// Split by the separator
const files = content.split(/\/\/ ==================== (.*?) ====================/);

// Process each file
for (let i = 1; i < files.length; i += 2) {
  const filePath = files[i].trim();
  const fileContent = files[i + 1].trim();
  
  // Create the full path
  const fullPath = path.join(__dirname, filePath);
  
  // Create directory if it doesn't exist
  fs.mkdirSync(path.dirname(fullPath), { recursive: true });
  
  // Write the file
  fs.writeFileSync(fullPath, fileContent);
  console.log(`Created: ${filePath}`);
}
