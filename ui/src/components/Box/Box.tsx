import React from "react";
import { motion, HTMLMotionProps } from "framer-motion";

export interface Props extends HTMLMotionProps<"div"> {
  children: React.ReactNode;
  direction?: "row" | "column";
  justify?: "center" | "flex-start" | "flex-end" | "space-between";
  align?: "center" | "flex-start" | "flex-end" | "space-between";
  duration?: number;
}

export const Box = ({
  children,
  direction = "column",
  justify = "center",
  align = "center",
  duration = 0.7,
  ...props
}: Props): JSX.Element => {
  return (
    <motion.div
      {...props}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration }}
      exit={{ opacity: 0 }}
      style={{
        display: "flex",
        flexDirection: direction,
        justifyContent: justify,
        alignItems: align,
        textAlign: "left",
      }}
    >
      {children}
    </motion.div>
  );
};
