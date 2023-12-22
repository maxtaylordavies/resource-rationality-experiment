import axios from "axios";

import { GRID_SIZE } from "./constants";

const api = axios.create({
  baseURL: "http://localhost:8002/api",
});

export const getRandomHeatmap = async () => {
  const response = await api.get(`/heatmap/random?size=${GRID_SIZE}&bins=4`);
  return response.data;
};

export const getHeatmapFromFile = async () => {
  const response = await api.get(`/heatmap/from_file`);
  return response.data as number[][];
};
