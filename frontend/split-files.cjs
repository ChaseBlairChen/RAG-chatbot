const fs = require('fs');
const path = require('path');

// Read the combined file
const content = fs.readFileSync('all_files_combined.txt', 'utf8');

// Split by the separator pattern
const sections = content.split(/\/\/ ==================== (.*?) ====================/);

// Process each file
for (let i = 1; i < sections.length; i += 2) {
  const filePath = sections[i].trim();
  const fileContent = sections[i + 1].trim();
  
  // Create the full path
  const fullPath = path.join(__dirname, filePath);
  
  // Create directory if it doesn't exist
  fs.mkdirSync(path.dirname(fullPath), { recursive: true });
  
  // Write the file
  fs.writeFileSync(fullPath, fileContent);
  console.log(`Created: ${filePath}`);
}

console.log('All files created successfully!');
