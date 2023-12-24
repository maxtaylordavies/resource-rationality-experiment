export const getValueFromUrlOrLocalstorage = (key: string) => {
  const params = new URLSearchParams(window.location.search);
  return (
    params.get(key) || JSON.parse(localStorage.getItem(`_rre_${key}`) || "null")
  );
};

export const writeToLocalStorage = (key: string, val: any) => {
  localStorage.setItem(`_rre_${key}`, JSON.stringify(val));
};

export const removeFromLocalStorage = (key: string) => {
  localStorage.removeItem(`_rre_${key}`);
};
