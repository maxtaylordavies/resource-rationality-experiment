import React from "react";
import { motion, MotionProps } from "framer-motion";

import "./button.css";

interface Props extends MotionProps {
  label: string;
  onClick: () => void;
}

export const Button = ({ label, onClick }: Props) => {
  return (
    <motion.button
      className="button"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onClick}
    >
      {label}
    </motion.button>
  );
};
