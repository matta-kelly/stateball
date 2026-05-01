import SimEvals from "./SimEvals";
import { CalibrationRuns } from "./CalibrationPage";

export default function WorkshopEvals() {
  return (
    <div className="space-y-8">
      <SimEvals />
      <CalibrationRuns />
    </div>
  );
}
