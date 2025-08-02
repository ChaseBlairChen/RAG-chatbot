// services/api.ts
export class ApiService {
  private baseUrl: string;
  private apiToken: string;

  constructor(baseUrl: string, apiToken: string) {
    this.baseUrl = baseUrl;
    this.apiToken = apiToken;
  }

  // ... existing methods ...

  // ADD THIS METHOD
  async searchFreeLegalDatabases(query: string): Promise<any> {
    return this.post('/external/search-free', { query });
  }
}

// Add these functions to your existing api.ts file

export const searchFreeLegalDatabases = async (query: string) => {
  const formData = new URLSearchParams();
  formData.append('query', query);
  
  const response = await fetch(`${DEFAULT_BACKEND_URL}/api/external/search-free`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      ...getAuthHeaders() // if you have auth headers
    },
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }
  
  return response.json();
};

export const searchLegalDatabases = async (query: string, databases: string[]) => {
  const formData = new URLSearchParams();
  formData.append('query', query);
  databases.forEach(db => formData.append('databases', db));
  
  const response = await fetch(`${DEFAULT_BACKEND_URL}/api/external/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      ...getAuthHeaders()
    },
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }
  
  return response.json();
};
