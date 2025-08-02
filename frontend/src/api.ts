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
