package main

import (
	"database/sql"
	"encoding/json"
	"math/rand"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

type Datastore struct {
	DB *sql.DB
}

type Heatmap [][]int

type ProlificMetadata struct {
	PRLFC_PID      string `json:"PRLFC_PID"`
	PRLFC_STUDY_ID string `json:"PRLFC_STUDY_ID"`
	PRLFC_SESS_ID  string `json:"PRLFC_SESS_ID"`
}

type Session struct {
	ID               int              `json:"id"`
	ExperimentId     string           `json:"experiment_id"`
	UserId           string           `json:"user_id"`
	CreatedAt        time.Time        `json:"created_at"`
	Texture          string           `json:"texture"`
	Cost             float64          `json:"cost"`
	Beta             float64          `json:"beta"`
	FinalScore       int              `json:"final_score"`
	TextResponse     string           `json:"text_response"`
	ProlificMetadata ProlificMetadata `json:"prolific_metadata"`
}

type Pos struct {
	Row int `json:"row"`
	Col int `json:"col"`
}

type ChoiceResult struct {
	Choice        []Pos `json:"choice"`         // list of two positions
	AgentSelected int   `json:"agent_selected"` // index of selected position (by agent) (i.e. 0 or 1)
	Selected      int   `json:"selected"`       // index of predicted position (by participant) (i.e. 0 or 1)
}

func CreateDatastore(dbPath string) (Datastore, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return Datastore{}, err
	}

	return Datastore{DB: db}, nil
}

func (ds *Datastore) Close() error {
	return ds.DB.Close()
}

func (ds *Datastore) RandomHeatmap(size int, bins int) Heatmap {
	// create a random 2D array of shape (size, size), with values in range [0, bins)
	heatmap := make(Heatmap, size)
	for i := range heatmap {
		heatmap[i] = make([]int, size)
		for j := range heatmap[i] {
			heatmap[i][j] = rand.Intn(bins)
		}
	}
	return heatmap
}

func (ds *Datastore) GetAllSessions() ([]Session, error) {
	rows, err := ds.DB.Query("SELECT id, experiment_id, user_id, created_at, texture, cost, beta, final_score FROM sessions")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sessions []Session
	for rows.Next() {
		var session Session
		err := rows.Scan(
			&session.ID,
			&session.ExperimentId,
			&session.UserId,
			&session.CreatedAt,
			&session.Texture,
			&session.Cost,
			&session.Beta,
			&session.FinalScore,
		)
		if err != nil {
			return nil, err
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

func (ds *Datastore) GetSession(id int) (Session, error) {
	var session Session
	var pmdStr string
	err := ds.DB.QueryRow("SELECT id, experiment_id, user_id, created_at, texture, cost, beta, final_score, text_response, prolific_metadata FROM sessions WHERE id=?", id).Scan(
		&session.ID,
		&session.ExperimentId,
		&session.UserId,
		&session.CreatedAt,
		&session.Texture,
		&session.Cost,
		&session.Beta,
		&session.FinalScore,
		&session.TextResponse,
		&pmdStr,
	)
	if err != nil {
		return Session{}, err
	}

	err = json.Unmarshal([]byte(pmdStr), &session.ProlificMetadata)
	if err != nil {
		return Session{}, err
	}

	return session, nil
}

func (ds *Datastore) CreateSession(experimentId string, userId string, cost float64, beta float64, prolificData ProlificMetadata) (Session, error) {
	stmt, err := ds.DB.Prepare("INSERT INTO sessions(experiment_id, user_id, created_at, texture, cost, beta, final_score, text_response, prolific_metadata) values(?, ?, ?, ?, ?, ?, ?, ?, ?)")
	if err != nil {
		return Session{}, err
	}
	defer stmt.Close()

	pmdBytes, err := json.Marshal(prolificData)
	if err != nil {
		return Session{}, err
	}
	pmdStr := string(pmdBytes)

	// for experiment 2 we don't vary texture
	texture := "smooth"

	createdAt := time.Now()
	res, err := stmt.Exec(
		experimentId,
		userId,
		createdAt,
		texture,
		cost,
		beta,
		0,
		"",
		pmdStr,
	)
	if err != nil {
		return Session{}, err
	}

	id, err := res.LastInsertId()
	if err != nil {
		return Session{}, err
	}

	// // set texture based on id: "rough" if odd, "smooth" if even
	// texture := "smooth"
	// if id%2 == 1 {
	// 	texture = "rough"
	// }
	// err = ds.SetSessionTexture(int(id), texture)
	// if err != nil {
	// 	return Session{}, err
	// }

	return Session{
		ID:               int(id),
		ExperimentId:     experimentId,
		UserId:           userId,
		CreatedAt:        createdAt,
		Texture:          texture,
		Cost:             cost,
		Beta:             beta,
		FinalScore:       0,
		TextResponse:     "",
		ProlificMetadata: prolificData,
	}, nil
}

func (ds *Datastore) SetSessionTexture(sessionId int, texture string) error {
	stmt, err := ds.DB.Prepare("UPDATE sessions SET texture=? WHERE id=?")
	if err != nil {
		return err
	}
	defer stmt.Close()

	_, err = stmt.Exec(texture, sessionId)
	return err
}

func (ds *Datastore) UpdateSession(sessionId int, finalScore int, textResponse string) error {
	stmt, err := ds.DB.Prepare("UPDATE sessions SET final_score=?, text_response=? WHERE id=?")
	if err != nil {
		return err
	}
	defer stmt.Close()

	_, err = stmt.Exec(finalScore, textResponse, sessionId)
	return err
}

func (ds *Datastore) RecordChoiceResult(sessionId int, round int, patchSize int, choiceResult ChoiceResult) error {
	stmt, err := ds.DB.Prepare("INSERT INTO choices(session_id, round, patch_size, row1, col1, row2, col2, agent_selected, selected) values(?, ?, ?, ?, ?, ?, ?, ?, ?)")
	if err != nil {
		return err
	}
	defer stmt.Close()

	_, err = stmt.Exec(
		sessionId,
		round,
		patchSize,
		choiceResult.Choice[0].Row,
		choiceResult.Choice[0].Col,
		choiceResult.Choice[1].Row,
		choiceResult.Choice[1].Col,
		choiceResult.AgentSelected,
		choiceResult.Selected,
	)

	return err
}
