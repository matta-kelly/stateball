import { Routes, Route, Navigate, useParams } from "react-router-dom";
import AuthGuard from "./components/AuthGuard";
import OperationsLayout from "./components/OperationsLayout";
import WorkshopLayout from "./components/WorkshopLayout";
import StatsLayout from "./components/StatsLayout";
import Workshop from "./pages/modeling/Workshop";
import EvalAnalysis from "./pages/modeling/EvalAnalysis";
import LiveGame from "./pages/LiveGame";
import GameDetail from "./pages/GameDetail";
import FeedMonitor from "./pages/FeedMonitor";
import Stats from "./pages/Stats";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Invite from "./pages/Invite";
import Admin from "./pages/Admin";

function RedirectGamePk() {
  const { gamePk } = useParams();
  return <Navigate to={`/game/${gamePk}`} replace />;
}

export default function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/invite/:token" element={<Invite />} />

      {/* Operations mode — full bleed, top bar */}
      <Route element={<AuthGuard><OperationsLayout /></AuthGuard>}>
        <Route index element={<LiveGame />} />
        <Route path="game/:gamePk" element={<GameDetail />} />
        <Route path="feed-monitor" element={<FeedMonitor />} />
        <Route path="admin" element={<Admin />} />
      </Route>

      {/* Workshop mode — unified page */}
      <Route element={<AuthGuard><WorkshopLayout /></AuthGuard>}>
        <Route path="workshop" element={<Workshop />} />
        <Route path="workshop/eval-analysis" element={<EvalAnalysis />} />
      </Route>

      {/* Stats mode — data verification */}
      <Route element={<AuthGuard><StatsLayout /></AuthGuard>}>
        <Route path="stats" element={<Stats />} />
      </Route>

      {/* Legacy redirects */}
      <Route path="/modeling" element={<Navigate to="/workshop" replace />} />
      <Route path="/modeling/*" element={<Navigate to="/workshop" replace />} />
      <Route path="/models/*" element={<Navigate to="/workshop" replace />} />
      <Route path="/live/:gamePk" element={<RedirectGamePk />} />
      <Route path="/live" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
