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

  askAssistant(question: string, history: any[] = [], sessionId: string | null = null): Observable<any> {
    return this.http.post(`${this.apiUrl}/ask-assistant`, {
      question,
      conversation_history: history,
      session_id: sessionId
    }, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
}
