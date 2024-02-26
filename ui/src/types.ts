export type Session = {
  id: number;
  experiment_id: string;
  user_id: string;
  created_at: Date;
  texture: string;
  cost: number;
  beta: number;
  final_score: number;
};

export type Heatmap = number[][];

export type Pos = {
  row: number;
  col: number;
};

export type ChoiceResult = {
  choice: Pos[];
  selected: number;
};
