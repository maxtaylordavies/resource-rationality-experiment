import React from "react";
import { MotionProps, motion } from "framer-motion";

import "./button.css";

// extend motion button props
interface Props extends MotionProps {
  label: string;
  to: string;
}

export const LinkButton = ({ label, to, ...props }: Props) => {
  return (
    <motion.button
      className="button"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => (window.location.href = to)}
      {...props}
    >
      {label}
    </motion.button>
  );
};
