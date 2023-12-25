import axios from "axios";

import { Session, Heatmap, ChoiceResult } from "./store";

const api = axios.create({
  baseURL: `${window.location.protocol}//${window.location.host}/api`,
});

export const createSession = async (
  experimentId: string,
  userId: string
): Promise<Session> => {
  const response = await api.post("/sessions/create", {
    experiment_id: experimentId,
    user_id: userId,
  });
  return response.data;
};

// export const getRandomHeatmap = async (): Promise<Heatmap> => {
//   const response = await api.get(`/heatmap/random?size=${GRID_SIZE}&bins=4`);
//   return response.data;
// };

export const getHeatmapFromFile = async (id: string): Promise<Heatmap> => {
  const response = await api.get(`/heatmap/from_file?id=${id}`);
  return response.data;
};

export const recordChoiceResult = async (
  sessionId: number,
  choiceResult: ChoiceResult
): Promise<void> => {
  await api.post("/choices/record", {
    session_id: sessionId,
    choice_result: choiceResult,
  });
};
