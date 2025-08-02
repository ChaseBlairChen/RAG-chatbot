// utils/fileValidation.ts
export const validateFileBeforeUpload = (file: File): string | null => {
  const maxSize = 50 * 1024 * 1024; // 50MB
  const allowedTypes = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
    'application/rtf'
  ];
  
  if (file.size > maxSize) {
    return `File "${file.name}" is too large: ${(file.size / 1024 / 1024).toFixed(2)}MB. Maximum allowed: 50MB`;
  }
  
  if (!allowedTypes.includes(file.type)) {
    const extension = '.' + (file.name.split('.').pop()?.toLowerCase() || '');
    return `File "${file.name}" has unsupported type: ${extension}. Allowed: PDF, DOC, DOCX, RTF, TXT`;
  }
  
  return null; // No errors
};
