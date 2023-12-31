package main

import (
	"database/sql"
	"math/rand"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

type Datastore struct {
	DB *sql.DB
}

type Heatmap [][]int

type Session struct {
	ID           int       `json:"id"`
	ExperimentId string    `json:"experiment_id"`
	UserId       string    `json:"user_id"`
	CreatedAt    time.Time `json:"created_at"`
}

type Pos struct {
	Row int `json:"row"`
	Col int `json:"col"`
}

type ChoiceResult struct {
	Choice   []Pos `json:"choice"`   // list of two positions
	Selected int   `json:"selected"` // index of selected position (i.e. 0 or 1)
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
	rows, err := ds.DB.Query("SELECT * FROM sessions")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sessions []Session
	for rows.Next() {
		var session Session
		err := rows.Scan(&session.ID, &session.ExperimentId, &session.UserId, &session.CreatedAt)
		if err != nil {
			return nil, err
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

func (ds *Datastore) CreateSession(experimentId string, userId string) (Session, error) {
	stmt, err := ds.DB.Prepare("INSERT INTO sessions(experiment_id, user_id, created_at) values(?, ?, ?)")
	if err != nil {
		return Session{}, err
	}
	defer stmt.Close()

	res, err := stmt.Exec(experimentId, userId, time.Now())
	if err != nil {
		return Session{}, err
	}

	id, err := res.LastInsertId()
	if err != nil {
		return Session{}, err
	}

	return Session{
		ID:           int(id),
		ExperimentId: experimentId,
		UserId:       userId,
		CreatedAt:    time.Now(),
	}, nil
}

func (ds *Datastore) RecordChoiceResult(sessionId int, choiceResult ChoiceResult) error {
	stmt, err := ds.DB.Prepare("INSERT INTO choices(session_id, row1, col1, row2, col2, selected) values(?, ?, ?, ?, ?, ?)")
	if err != nil {
		return err
	}
	defer stmt.Close()

	_, err = stmt.Exec(
		sessionId,
		choiceResult.Choice[0].Row,
		choiceResult.Choice[0].Col,
		choiceResult.Choice[1].Row,
		choiceResult.Choice[1].Col,
		choiceResult.Selected,
	)

	return err
}
