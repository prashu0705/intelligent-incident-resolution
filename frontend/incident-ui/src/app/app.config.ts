import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router'; // Needed for appRoutes if any, good practice
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideAnimations } from '@angular/platform-browser/animations';
// import { provideBrowserGlobalErrorListeners } from '@angular/core'; // Optional, can keep if needed

import { routes } from './app.routes'; // Assuming you have app.routes.ts for routing

export const appConfig: ApplicationConfig = {
  providers: [
    // provideBrowserGlobalErrorListeners(), // Optional
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes), // Configure routing
    provideHttpClient(withFetch()),      // Provide HttpClient with fetch API
    provideAnimations()       // Provide animations for Angular Material
  ]
};
