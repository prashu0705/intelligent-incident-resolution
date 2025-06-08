import { Routes } from '@angular/router';
import { AssistantChatComponent } from './components/assistant-chat/assistant-chat';
import { TicketListComponent } from './components/ticket-list/ticket-list';
import { ResolutionViewerComponent } from './components/resolution-viewer/resolution-viewer';

export const routes: Routes = [
  { path: '', component: AssistantChatComponent },
  { path: 'tickets', component: TicketListComponent },
  { path: 'resolution', component: ResolutionViewerComponent },
];

