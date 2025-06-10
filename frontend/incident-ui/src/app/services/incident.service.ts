import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class IncidentService {
  private apiUrl = 'http://localhost:8000'; // Your FastAPI port

  constructor(private http: HttpClient) {}

  searchSimilarIncidents(query: string, top_k: number = 5): Observable<any> {
    return this.http.post(`${this.apiUrl}/search-similar-incidents`, {
      query,
      top_k
    });
  }

  recommendResolution(query: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/recommend-resolution`, {
      query,
      top_k: 3
    });
  }

  askAssistant(
    question: string,
    conversationHistory: Array<{role: string, content: string}> = [],
    sessionId: string | null = null
  ): Observable<any> {
    let headers: {[header: string]: string | string[]} = {
      'Content-Type': 'application/json'
    };

    if (sessionId) {
      headers['X-Session-Id'] = sessionId;
    }

    return this.http.post(`${this.apiUrl}/ask-assistant`, {
      question,
      conversation_history: conversationHistory // Ensure this matches backend Pydantic model
    }, { headers });
  }
}
