import axios from "axios";

import { Session, Heatmap, ChoiceResult } from "./store";

const api = axios.create({
  baseURL: `${window.location.protocol}//${window.location.host}/api`,
});

export const createSession = async (
  experimentId: string,
  userId: string,
  texture: string,
  cost: number,
  prolificMetadata: { [key: string]: string },
): Promise<Session> => {
  const response = await api.post("/sessions/create", {
    experiment_id: experimentId,
    user_id: userId,
    texture,
    cost,
    prolific_metadata: prolificMetadata,
  });
  return response.data;
};

export const getSession = async (sessionId: number): Promise<Session> => {
  const response = await api.get(`/sessions/get?id=${sessionId}`);
  return response.data;
};

export const updateSession = async (
  sessionId: number,
  finalScore: number,
  textResponse: string,
) => {
  await api.post("/sessions/update", {
    id: sessionId,
    final_score: finalScore,
    text_response: textResponse,
  });
};

export const getTutorialHeatmap = async (): Promise<Heatmap> => {
  const response = await api.get("/heatmap/tutorial");
  return response.data;
};

export const getHeatmapFromFile = async (
  texture: string,
  round: string | number,
  patchSize: number,
): Promise<Heatmap> => {
  const response = await api.get(
    `/heatmap/from_file?texture=${texture}&round=${round}&ps=${patchSize}`,
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
