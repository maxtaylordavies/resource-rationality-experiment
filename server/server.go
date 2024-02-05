package main

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"
	"log"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
)

const (
	dbPath      = "store.db"
	uiPath      = "ui"
	distPath    = uiPath + "/dist"
	indexPath   = distPath + "/index.html"
	logoPath    = uiPath + "/public/logo.png"
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
	Store  Datastore
}

func CreateServer() (Server, error) {
	s := Server{
		Port: ":8101",
	}

	// initiate datastore
	store, err := CreateDatastore(dbPath)
	if err != nil {
		return s, err
	}
	s.Store = store

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

	// routes for serving api
	s.Router.HandleFunc("/api/heatmap/random", func(w http.ResponseWriter, r *http.Request) {
		sizeStr := r.URL.Query().Get("size")
		size, err := strconv.Atoi(sizeStr)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		binsStr := r.URL.Query().Get("bins")
		bins, err := strconv.Atoi(binsStr)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		heatmap := s.Store.RandomHeatmap(size, bins)
		respond(w, heatmap)
	}).Methods("GET")

	s.Router.HandleFunc("/api/heatmap/from_file", func(w http.ResponseWriter, r *http.Request) {
		round := r.URL.Query().Get("round")
		ps := r.URL.Query().Get("ps")
		if round == "" || ps == "" {
			http.Error(w, "missing round or ps", http.StatusBadRequest)
			return
		}

		http.ServeFile(w, r, "heatmaps/" + round + "/" + ps + ".txt")
	}).Methods("GET")

	s.Router.HandleFunc("/api/sessions/all", func(w http.ResponseWriter, r *http.Request) {
		sessions, err := s.Store.GetAllSessions()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		respond(w, sessions)
	}).Methods("GET")

	s.Router.HandleFunc("/api/sessions/get", func(w http.ResponseWriter, r *http.Request) {
		idStr := r.URL.Query().Get("id")
		id, err := strconv.Atoi(idStr)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		session, err := s.Store.GetSession(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		respond(w, session)
	}).Methods("GET")

	s.Router.HandleFunc("/api/sessions/create", func(w http.ResponseWriter, r *http.Request) {
		log.Println("/api/sessions/create")
		decoder := json.NewDecoder(r.Body)
		var payload struct {
			ExperimentId string `json:"experiment_id"`
			UserId       string `json:"user_id"`
			ChoiceReward int    `json:"choice_reward"`
		}

		err := decoder.Decode(&payload)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		log.Println(payload)

		session, err := s.Store.CreateSession(payload.ExperimentId, payload.UserId, payload.ChoiceReward)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		respond(w, session)
	}).Methods("POST")

	s.Router.HandleFunc("/api/choices/record", func(w http.ResponseWriter, r *http.Request) {
		decoder := json.NewDecoder(r.Body)
		var payload struct {
			SessionId    int          `json:"session_id"`
			ChoiceResult ChoiceResult `json:"choice_result"`
		}

		err := decoder.Decode(&payload)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		err = s.Store.RecordChoiceResult(payload.SessionId, payload.ChoiceResult)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}).Methods("POST")

	// assets
	s.Router.PathPrefix("/assets").HandlerFunc(dirHandlerFunc(uiPath, false))

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
