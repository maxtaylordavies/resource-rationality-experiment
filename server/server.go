package main

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
)

const (
	dbPath      = "jikji.db"
	distPath    = "ui/dist"
	indexPath   = distPath + "/index.html"
	logoPath    = "ui/public/logo.png"
	faviconPath = distPath + "/favicon.ico"
)

func dirHandlerFunc(path string, longLived bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if longLived {
			w.Header().Set("Cache-Control", "public, max-age=31536000")
		}
		http.FileServer(http.Dir(path)).ServeHTTP(w, r)
	}
}

func fileHandlerFunc(filePath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, filePath)
	}
}

func respond(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

type Server struct {
	Router *mux.Router
	Port   string
}

func CreateServer() (Server, error) {
	s := Server{
		Port: ":8002",
	}

	// initiate router and return server
	s.registerRoutes()
	return s, nil
}

func (s *Server) registerRoutes() {
	s.Router = mux.NewRouter()

	// for uptime checker
	s.Router.HandleFunc("/status", func(w http.ResponseWriter, r *http.Request) {
		payload := struct {
			Status string `json:"status"`
		}{
			Status: "ok",
		}
		respond(w, payload)
	})

	// routes for serving ui
	s.Router.PathPrefix("/static").HandlerFunc(dirHandlerFunc(distPath, false))
	for _, p := range []string{"images", "css", "fonts"} { // things we want client to cache with long expiration
		s.Router.PathPrefix("/" + p).HandlerFunc(dirHandlerFunc(distPath, true))
	}
	s.Router.HandleFunc("/logo.png", fileHandlerFunc(logoPath))
	s.Router.HandleFunc("/favicon.ico", fileHandlerFunc(faviconPath))
	s.Router.PathPrefix("/").HandlerFunc(fileHandlerFunc(indexPath))
}

func (s Server) ListenAndServe() error {
	srv := http.Server{
		Addr:         s.Port,
		Handler:      handlers.CORS()(s.Router),
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	return srv.ListenAndServe()
}
