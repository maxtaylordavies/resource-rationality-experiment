import axios from "axios";

import { Session, Heatmap, ChoiceResult } from "./store";

const api = axios.create({
  baseURL: `${window.location.protocol}//${window.location.host}/api`,
});

export const createSession = async (
  experimentId: string,
  userId: string,
  choiceReward = 10,
): Promise<Session> => {
  const response = await api.post("/sessions/create", {
    experiment_id: experimentId,
    user_id: userId,
    choice_reward: choiceReward,
  });
  return response.data;
};

export const getSession = async (sessionId: number): Promise<Session> => {
  const response = await api.get(`/sessions/get?id=${sessionId}`);
  return response.data;
};

export const getHeatmapFromFile = async (
  round: string | number,
  patchSize: number,
): Promise<Heatmap> => {
  const response = await api.get(
    `/heatmap/from_file?round=${round}&ps=${patchSize}`,
  );
  return response.data;
};

export const recordChoiceResult = async (
  sessionId: number,
  round: number,
  patchSize: number,
  choiceResult: ChoiceResult,
): Promise<void> => {
  await api.post("/choices/record", {
    session_id: sessionId,
    round,
    patch_size: patchSize,
    choice_result: choiceResult,
  });
};
