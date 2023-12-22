package main

import (
	"database/sql"
	"math/rand"
)

type Datastore struct {
	DB *sql.DB
}

type Heatmap [][]int

func CreateDatastore(dbPath string) (Datastore, error) {
	// db, err := sql.Open("sqlite3", dbPath)
	// if err != nil {
	// 	return Datastore{}, err
	// }

	// return Datastore{DB: db}, nil
	return Datastore{}, nil
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
