import React from 'react';
import Home from './pages/Home';
import { Routes, Route } from 'react-router-dom';
import TripDashboard from './pages/TripDashboard';
import { CssBaseline } from '@mui/material';

const App = () => {
  return (
    <>
      <CssBaseline />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<TripDashboard />} />
      </Routes>
    </>
  );
};

export default App;
