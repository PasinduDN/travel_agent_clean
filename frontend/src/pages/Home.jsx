import React, { useState, useMemo } from 'react';
import { Container, Paper, TextareaAutosize } from '@mui/material';
import { useNavigate } from 'react-router-dom'; // 1. Import useNavigate
import ButtonGroup from '@mui/material/ButtonGroup';
import {
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Typography,
  Snackbar
} from "@mui/material";

import Alert from "@mui/material/Alert";
import { LocalizationProvider, DatePicker } from "@mui/x-date-pickers";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { CircularProgress } from "@mui/material";
import { DateCalendar } from '@mui/x-date-pickers/DateCalendar';
import travelPreview from '../assets/travel-preview.png';
import Checkbox from "@mui/material/Checkbox";
import ListItemText from "@mui/material/ListItemText";

const Home = () => {
  const navigate = useNavigate();
  // const [chatContent, setChatContent] = useState('Hello! How can I assist you with your travel plans?');
  const [language, setLanguage] = useState('English');
  const [form, setForm] = useState("");
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const buttonColor = '#056d96ff';
  const buttonHover = '#054274ff';

  // Location & Date State
  const [location, setLocation] = useState("");
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [openLocationModal, setOpenLocationModal] = useState(false);
  const [openDateModal, setOpenDateModal] = useState(false);
  const [warningOpen, setWarningOpen] = useState(false);

  const [locations, setLocations] = useState([]);

  // const locations = ["Colombo", "Kandy", "Galle", "Nuwara Eliya", "Ella", "Sigiriya"];
  const [errorMessage, setErrorMessage] = useState("");

  const [openWizard, setOpenWizard] = useState(false);
  const [step, setStep] = useState(1);

  const handleNext = () => setStep(prev => Math.min(prev + 1, 7));
  const handlePrev = () => setStep(prev => Math.max(prev - 1, 1));


  const [chatHistory, setChatHistory] = useState([
    { sender: "AI", message: "Hello! How can I assist you with your travel plans?" }
  ]);
  const sessionId = useMemo(() => `session-${Date.now()}`, []);

  function ExtentQuestion(newText) {
    setQuery(prevQuery => {
      if (!prevQuery) return newText;
      const trimmed = prevQuery.trim();
      if (trimmed.endsWith('.')) return trimmed + ' ' + newText;
      return trimmed + '. ' + newText;
    });
  }

  const handlePreferencesSubmit = async () => {
    setOpenWizard(false)

    if (!locations || !startDate || !endDate) {
      setWarningOpen(true);
      return;
    }
    setIsLoading(true);

    try {

      const response = await fetch(`http://127.0.0.1:8000/api/process_preferences/${sessionId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          preferences_text: query,
          locations: locations,
          start_date: startDate.format("YYYY-MM-DD"),
          end_date: endDate.format("YYYY-MM-DD"),
        }),
      });


      const data = await response.json();
      // setChatContent(data.result); // display the AI output
      console.log("data.posts", data.posts);
      

      // Then after response
      setChatHistory(prev => [
        ...prev,
        { sender: "User", message: query },
        { sender: "AI", message: data.result }
      ]);
      setQuery('');

      // Navigate to the dashboard and pass the API result in the state
      navigate('/dashboard', {
        state: {
          tripData: data.result,
          query: query,
          locations: locations,   // ✅ plural, matches backend
          start_date: startDate.format("YYYY-MM-DD"),
          end_date: endDate.format("YYYY-MM-DD")
        }
      });

    } catch (error) {
      setErrorMessage("Error connecting to backend.");
    } finally {
      // Hide loading indicator once the request is finished
      setIsLoading(false);
    }
  };

  const chatWithAgent = async () => {

    const chatWithAgentOption = 1
    const defaultTripData = "Let's start planning your trip!";
    const defaultLocations = []; // Or an empty array, depending on your preference
    const defaultStartDate = "";  // Default start date
    const defaultEndDate = "";    // Default end date
    const defaultQuery = "";

    navigate('/dashboard', {
      state: {
        tripData: defaultTripData,  // Placeholder message
        query: defaultQuery,
        locations: defaultLocations,  // Use default locations
        start_date: defaultStartDate,
        end_date: defaultEndDate,
        chatWithAgentOption: chatWithAgentOption
      }
    });
  };

  const districts = [
    "Colombo",
    "Gampaha",
    "Kalutara",
    "Kandy",
    "Matale",
    "Nuwara Eliya",
    "Galle",
    "Matara",
    "Hambantota",
    "Jaffna",
    "Kilinochchi",
    "Mannar",
    "Vavuniya",
    "Mullaitivu",
    "Batticaloa",
    "Ampara",
    "Trincomalee",
    "Kurunegala",
    "Puttalam",
    "Anuradhapura",
    "Polonnaruwa",
    "Badulla",
    "Monaragala",
    "Ratnapura",
    "Kegalle"
  ];

  return (
    <Container
      maxWidth="xl"
      sx={{
        color: 'var(--primary-text-color)',
        paddingY: 4
      }}
    >
      <>
        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <CircularProgress />
          </Box>
        ) : (

          <>
            <Box className="hero">
              {/* Left: Text Content */}
              <Box className="hero-text">
                <Typography variant="overline" color="var(--accent-color)">
                  Smarter Journeys, Less Planning
                </Typography>

                <h1>
                  Design Your <span className="accent">Perfect Escape</span> with Ease
                </h1>

                <p>
                  Skip the hours of research. Our AI suggests destinations, stays, and
                  experiences that match your mood, so every trip feels effortless.
                </p>

                {/* Key Points */}
                <Box className="hero-points">
                  <span className="point">⚡ Fast Suggestions</span>
                  <span className="point">🌍 Local Insights</span>
                  <span className="point">💰 Budget Aware</span>
                  <span className="point">📱 Works on Any Device</span>
                </Box>

                {/* CTA Button */}
                <button onClick={() => setOpenWizard(true)} className="hero-cta">Start Exploring →</button>
                {/* <button onClick={chatWithAgent} className="hero-cta-ai">Chat with Agent →</button> */}

                <Box mt={2}>
                  <Typography variant="body2" color="var(--accent-color)">
                    ★★★★★ Loved by thousands of travelers
                  </Typography>
                </Box>
              </Box>

              {/* Right: Image Preview */}
              <Box className="hero-image">
                <img src={travelPreview} alt="Trip Planner Preview" />
              </Box>
            </Box>


            <Box sx={{ marginTop: 3 }}>
              {/* <Typography variant="h5" gutterBottom>Ask me about your travel preferences!</Typography> */}
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  '& > *': {
                    m: 1,
                  },
                }}
              >
                {/* --- Location & Date Buttons --- */}
                {/* <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <Button variant="contained" onClick={() => setOpenLocationModal(true)}>Select Location</Button>
            <Button variant="contained" onClick={() => setOpenDateModal(true)}>Select Travel Dates</Button>
          </Box> */}
                {/* 
          <Typography>Selected Location: {location || "None"}</Typography>
          <Typography>
            Selected Dates: {startDate ? startDate.format("YYYY-MM-DD") : "None"} - {endDate ? endDate.format("YYYY-MM-DD") : "None"}
          </Typography> */}

                {/* --- Location Modal --- */}
                {/* <Dialog open={openLocationModal} onClose={() => setOpenLocationModal(false)}>
            <DialogTitle>Select Location</DialogTitle>
            <DialogContent>
              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel>Location</InputLabel>
                <Select
                  value={location}
                  label="Location"
                  onChange={(e) => setLocation(e.target.value)}
                >
                  {locations.map((loc) => (
                    <MenuItem key={loc} value={loc}>{loc}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setOpenLocationModal(false)}>Close</Button>
            </DialogActions>
          </Dialog> */}

                {/* --- Date Modal --- */}
                {/* <Dialog open={openDateModal} onClose={() => setOpenDateModal(false)}>
            <DialogTitle>Select Travel Dates</DialogTitle>
            <DialogContent sx={{ display: 'flex', gap: 2, mt: 1 }}>
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <DatePicker
                  label="Start Date"
                  value={startDate}
                  onChange={(newValue) => setStartDate(newValue)}
                  renderInput={(params) => <TextField {...params} />}
                />
                <DatePicker
                  label="End Date"
                  value={endDate}
                  onChange={(newValue) => setEndDate(newValue)}
                  renderInput={(params) => <TextField {...params} />}
                />
              </LocalizationProvider>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setOpenDateModal(false)}>Close</Button>
            </DialogActions>
          </Dialog> */}

                <Dialog
                  open={openWizard}
                  onClose={() => setOpenWizard(false)}
                  fullWidth
                  maxWidth="md"
                  // Make the entire dialog (Paper) use ONE background (no white)
                  PaperProps={{
                    style: {
                      background: 'linear-gradient(135deg, var(--bg-gradient-start), var(--bg-gradient-end))',
                      color: 'var(--primary-text-color)',
                      borderRadius: 20,
                      boxShadow: '0 24px 80px rgba(0,0,0,0.55)',
                      overflow: 'hidden', // keeps rounded corners tidy
                    }
                  }}
                >
                  <div style={{
                    padding: '2px',
                    // background: 'transparent',
                    minHeight: 'auto',
                    maxHeight: 'calc(100vh - 120px)',  // Prevents the content from overflowing
                    overflowY: 'auto',  // Enables scroll for overflow
                    // display: 'flex',
                    flexDirection: 'column',
                    gap: 18
                  }}>
                    {/* Close button row (transparent) */}
                    <div style={{
                      display: 'flex',
                      justifyContent: 'flex-end',
                      padding: '10px 12px',
                      background: 'transparent'
                    }}>
                      <Button
                        onClick={() => setOpenWizard(false)}
                        style={{
                          minWidth: 36,
                          height: 36,
                          padding: 0,
                          borderRadius: 10,
                          color: 'white',
                          border: '1px solid rgba(255,255,255,0.25)',
                          background: 'rgba(255,255,255,0.06)'
                        }}
                      >
                        ✕
                      </Button>
                    </div>

                    {/* Stepper (transparent bar, no white) */}
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '12px 22px',
                      background: 'transparent',              // << important: no white
                      gap: 12,
                      flexWrap: 'wrap'
                    }}>
                      {[1, 2, 3, 4, 5, 6, 7].map((num, i, arr) => (
                        <div key={num} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <div style={{
                            width: 36, height: 36, borderRadius: '50%',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontWeight: 700, fontSize: 14,
                            background: step > num ? '#27ae60' : (step === num ? '#2575fc' : '#2b5c66'),
                            color: 'white',
                            boxShadow: step === num ? '0 0 0 3px rgba(37,117,252,0.25)' : 'none'
                          }}>
                            {step > num ? `${num}` : num}
                          </div>

                          {/* connector line, except after last dot */}
                          {i < arr.length - 1 && (
                            <div style={{
                              width: 70, height: 4, borderRadius: 2,
                              background: step > num ? 'linear-gradient(90deg, #2ddf85, #2ddf85)'
                                : 'linear-gradient(90deg, #1e6b73, #1e6b73)'
                            }} />
                          )}
                        </div>
                      ))}
                    </div>

                    {/* CONTENT AREA (kept one color; no inner white) */}
                    <div style={{
                      padding: '22px',
                      background: 'transparent',              // << important: no white band
                      minHeight: 320,                          // prevents jumping height
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 18
                    }}>

                      {/* Q1: Location */}
                      {step === 1 && (
                        <div style={{ display: "grid", gap: 12 }}>
                          <Typography variant="h6" style={{ color: "white" }}>
                            Select Your Location(s)
                          </Typography>

                          <FormControl fullWidth>
                            <InputLabel style={{ color: "rgba(255,255,255,0.85)" }}>
                              Locations
                            </InputLabel>
                            <Select
                              multiple
                              value={locations}
                              onChange={(e) => setLocations(e.target.value)}
                              renderValue={(selected) => selected.join(", ")}
                              style={{
                                borderRadius: 12,
                                background: "rgba(255,255,255,0.08)",
                                color: "white",
                              }}
                              MenuProps={{
                                PaperProps: { style: { background: "#0f2a31", color: "white" } },
                              }}
                            >
                              {districts.map((loc) => (
                                <MenuItem key={loc} value={loc}>
                                  <Checkbox checked={locations.indexOf(loc) > -1} />
                                  <ListItemText primary={loc} />
                                </MenuItem>
                              ))}
                            </Select>
                          </FormControl>
                        </div>
                      )}

                      {/* Q2: Two inline calendars (centered) */}
                      {step === 2 && (
                        <div style={{ display: 'grid', gap: 14 }}>
                          <Typography variant="h6" style={{ color: 'white' }}>Choose Your Travel Dates</Typography>

                          <LocalizationProvider dateAdapter={AdapterDayjs}>
                            <div
                              style={{
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'start',
                                gap: 24,
                                flexWrap: 'wrap',
                              }}
                            >
                              {/* Start */}
                              <div
                                style={{
                                  borderRadius: 16,
                                  // padding: 8,
                                  background: 'rgba(255,255,255,0.06)',
                                  border: '1px solid rgba(255,255,255,0.18)',
                                  width: '100%',
                                  maxWidth: '300px', // Limit the max width for large screens
                                }}
                              >
                                <Typography
                                  variant="body2"
                                  style={{
                                    color: 'rgba(255,255,255,0.85)',
                                    marginBottom: 8,
                                  }}
                                >
                                  Start
                                </Typography>
                                <DateCalendar
                                  value={startDate}
                                  onChange={(newValue) => setStartDate(newValue)}
                                  slotProps={{
                                    day: {
                                      sx: {
                                        color: 'white',
                                        '&.Mui-selected': {
                                          backgroundColor: '#2575fc',
                                          color: '#fff',
                                        },
                                        '&.MuiPickersDay-today': {
                                          border: '1px solid #00d4ff',
                                        },
                                      },
                                    },
                                    switchViewButton: { sx: { color: 'white' } },
                                    previousIconButton: { sx: { color: 'white' } },
                                    nextIconButton: { sx: { color: 'white' } },
                                  }}
                                />
                              </div>

                              {/* End */}
                              <div
                                style={{
                                  borderRadius: 16,
                                  // padding: 2,
                                  background: 'rgba(255,255,255,0.06)',
                                  border: '1px solid rgba(255,255,255,0.18)',
                                  width: '100%',
                                  maxWidth: '300px', // Limit the max width for large screens
                                }}
                              >
                                <Typography
                                  variant="body2"
                                  style={{
                                    color: 'rgba(255,255,255,0.85)',
                                    marginBottom: 8,
                                  }}
                                >
                                  End
                                </Typography>
                                <DateCalendar
                                  value={endDate}
                                  onChange={(newValue) => setEndDate(newValue)}
                                  slotProps={{
                                    day: {
                                      sx: {
                                        color: 'white',
                                        '&.Mui-selected': {
                                          backgroundColor: '#2575fc',
                                          color: '#fff',
                                        },
                                        '&.MuiPickersDay-today': {
                                          border: '1px solid #00d4ff',
                                        },
                                      },
                                    },
                                    switchViewButton: { sx: { color: 'white' } },
                                    previousIconButton: { sx: { color: 'white' } },
                                    nextIconButton: { sx: { color: 'white' } },
                                  }}
                                />
                              </div>
                            </div>
                          </LocalizationProvider>
                        </div>
                      )}


                      {/* Q3: Destination */}
                      {step === 3 && (
                        <div style={{ display: 'grid', gap: 12 }}>
                          <Typography variant="h6" style={{ color: 'white' }}>Destination Preferences</Typography>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                            {["Beach", "City", "Mountains", "Historical Sites", "Adventure", "Nature & Wildlife"].map(item => {
                              const picked = query.includes(item);
                              return (
                                <Button
                                  key={item}
                                  disabled={picked}
                                  onClick={() => ExtentQuestion(`I am interested in ${item}. `)}
                                  style={{
                                    flex: '1 1 45%',
                                    minWidth: 140,
                                    padding: '12px 14px',
                                    borderRadius: 12,
                                    background: picked ? '#27444a' : '#2575fc',
                                    color: 'white',
                                    textTransform: 'none'
                                  }}
                                >
                                  {item}
                                </Button>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Q4: Accommodation */}
                      {step === 4 && (
                        <div style={{ display: 'grid', gap: 12 }}>
                          <Typography variant="h6" style={{ color: 'white' }}>Accommodation Preferences</Typography>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(140px,1fr))', gap: 12 }}>
                            {["Business Travel", "Beach Resort", "Boutique Art Hotel", "Eco-Lodge", "Family-Friendly", "Budget-Friendly", "Luxury"].map(item => {
                              const picked = query.includes(item);
                              return (
                                <Button
                                  key={item}
                                  disabled={picked}
                                  onClick={() => ExtentQuestion(`I prefer ${item}. `)}
                                  style={{
                                    padding: '12px 14px',
                                    borderRadius: 12,
                                    background: picked ? '#274a36' : '#27ae60',
                                    color: 'white',
                                    textTransform: 'none'
                                  }}
                                >
                                  {item}
                                </Button>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Q5: Food */}
                      {step === 5 && (
                        <div style={{ display: 'grid', gap: 12 }}>
                          <Typography variant="h6" style={{ color: 'white' }}>Food Preferences</Typography>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                            {["Local Sri Lankan", "International", "Vegetarian", "Seafood", "Street Food"].map(item => {
                              const picked = query.includes(item);
                              return (
                                <Button
                                  key={item}
                                  disabled={picked}
                                  onClick={() => ExtentQuestion(`I prefer ${item} cuisine. `)}
                                  style={{
                                    flex: '1 1 45%',
                                    minWidth: 140,
                                    padding: '12px 14px',
                                    borderRadius: 12,
                                    background: picked ? '#4a3b34' : '#ff884d',
                                    color: 'white',
                                    textTransform: 'none'
                                  }}
                                >
                                  {item}
                                </Button>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Q6: Activities */}
                      {step === 6 && (
                        <div style={{ display: 'grid', gap: 12 }}>
                          <Typography variant="h6" style={{ color: 'white' }}>Activity Preferences</Typography>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                            {["Wildlife Safari", "Cultural Heritage", "Food & Culinary Tours", "Hiking & Nature Trails", "Water Sports", "Wellness"].map(item => {
                              const picked = query.includes(item);
                              return (
                                <Button
                                  key={item}
                                  disabled={picked}
                                  onClick={() => ExtentQuestion(`I am interested in ${item}. `)}
                                  style={{
                                    flex: '1 1 45%',
                                    minWidth: 140,
                                    padding: '12px 14px',
                                    borderRadius: 12,
                                    background: picked ? '#3c2e48' : '#8e44ad',
                                    color: 'white',
                                    textTransform: 'none'
                                  }}
                                >
                                  {item}
                                </Button>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Q7: Weather */}
                      {step === 7 && (
                        <div style={{ display: 'grid', gap: 12 }}>
                          <Typography variant="h6" style={{ color: 'white' }}>Weather Preferences</Typography>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 12 }}>
                            {["Sunny & Warm", "Mild & Cool", "Avoid Rainy Areas", "Cloudy & Dry", "Cooler Temperature", "Warmer Temperature", "Moderate Temperature"].map(item => {
                              const picked = query.includes(item);
                              return (
                                <Button
                                  key={item}
                                  disabled={picked}
                                  onClick={() => ExtentQuestion(`I prefer ${item} weather. `)}
                                  style={{
                                    padding: '12px 14px',
                                    borderRadius: 12,
                                    background: picked ? '#23313a' : '#34495e',
                                    color: 'white',
                                    textTransform: 'none'
                                  }}
                                >
                                  {item}
                                </Button>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* ANSWER FIELD (visible text, no white bar) */}
                    <div style={{ padding: '0 22px 12px', background: 'transparent' }}>
                      <TextField
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        multiline
                        rows={3}
                        placeholder="Your selected preferences will appear here. You can edit or delete text."
                        variant="outlined"
                        fullWidth
                        // style the Input container
                        InputProps={{
                          style: {
                            background: 'rgba(255,255,255,0.08)',
                            borderRadius: 12,
                            border: '1px solid rgba(0,212,255,0.45)',
                            padding: '8px 10px'
                          }
                        }}
                        // style the actual input text (this is what fixes 'text not visible')
                        inputProps={{
                          style: {
                            color: 'white',
                            lineHeight: 1.5
                          }
                        }}
                      />
                    </div>

                    {/* NAVIGATION (transparent; buttons only) */}
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '12px 22px 22px',
                      background: 'transparent' // << important: no white band
                    }}>
                      {step !== 1 && (
                        <Button
                          onClick={handlePrev}
                          disabled={step === 1}
                          style={{
                            border: '1px solid rgba(255,255,255,0.25)',
                            color: 'white',
                            borderRadius: 10,
                            background: 'rgba(255,255,255,0.06)',
                          }}
                          className="button-responsive"
                          sx={{
                            fontSize: {
                              xs: '0.7rem',  // Smaller font size for mobile (xs)
                              sm: '0.7rem',      // Normal font size for tablets (sm)
                              md: '0.8rem',    // Larger font size for desktops (md)
                            }
                          }}
                        >
                          ← Previous
                        </Button>
                      )}


                      {step < 7 ? (
                        (step === 1 && locations.length > 0) ||                 // Step 1: require location
                          (step === 2 && startDate && endDate) ||     // Step 2: require both dates
                          (step > 2) ?
                          (<Button
                            onClick={handleNext}
                            style={{
                              border: 'none',
                              color: 'white',
                              borderRadius: 10,
                              background: 'linear-gradient(135deg, #4688ebff, #058075ff)'
                            }}
                            className="button-responsive"
                            sx={{
                              fontSize: {
                                xs: '0.7rem',  // Smaller font size for mobile (xs)
                                sm: '0.7rem',      // Normal font size for tablets (sm)
                                md: '0.8rem',    // Larger font size for desktops (md)
                              }
                            }}
                          >
                            Next Step →
                          </Button>
                          ) : null
                      ) : (
                        <Button
                          onClick={handlePreferencesSubmit}
                          style={{
                            border: 'none',
                            color: '#052b22',
                            fontWeight: 700,
                            borderRadius: 10,
                            padding: '10px 18px',
                            background: 'linear-gradient(135deg, #43e97b, #38f9d7)'
                          }}
                        >
                          Finish ✔
                        </Button>
                      )}
                    </div>
                  </div>
                </Dialog>




                {/* --- Warning Snackbar --- */}
                <Snackbar
                  open={warningOpen}
                  autoHideDuration={3000}
                  onClose={() => setWarningOpen(false)}
                  anchorOrigin={{ vertical: "top", horizontal: "center" }}
                >
                  <Alert severity="warning" onClose={() => setWarningOpen(false)}>
                    Please select a location and travel dates before submitting!
                  </Alert>
                </Snackbar>
              </Box>

              {form !== "" && (
                form === "01" ? (
                  <Box sx={{ padding: 2, backgroundColor: '#f0f0f07c', borderRadius: 1, minHeight: '200px' }}>
                    {/* Destination Preferences */}
                    <Paper className="frosted-card" sx={{ padding: 2 }}>
                      <Typography variant="body1" color="text.primary" mb={1}>Destination Preferences</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Destination Preferences" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I am interested in visiting the Beach. ")}>Beach</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I am interested in exploring the City. ")}>City</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer visiting Mountains. ")}>Mountains</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to explore Historical Sites. ")}>Historical Sites</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I am interested in Adventure activities around the country. ")}>Adventure</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to visit Nature and Wildlife areas. ")}>Nature & Wildlife</Button>
                      </ButtonGroup>

                      {/* Sub-category 1 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Special Interests</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Special Interests" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I am interested in Adventure destinations. ")}>Adventure</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to visit Nature & Wildlife areas. ")}>Nature & Wildlife</Button>
                      </ButtonGroup>

                      {/* Sub-category 2 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Hidden Gems</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Hidden Gems" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to explore less-known villages. ")}>Villages</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to see offbeat natural spots. ")}>Offbeat Nature</Button>
                      </ButtonGroup>

                    </Paper>
                  </Box>
                ) : form === "02" ? (
                  <Box sx={{ padding: 2, backgroundColor: '#f0f0f0', borderRadius: 1, minHeight: '200px' }}>
                    {/* Accommodation Preferences */}
                    <Paper className="frosted-card" sx={{ padding: 2 }}>
                      <Typography variant="body1" color="text.primary" mb={1}>Accommodation Preferences</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Accommodation Preferences" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Business Travel hotels. ")}>Business Travel</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer a Beach Resort. ")}>Beach Resort</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer a Boutique Art Hotel. ")}>Boutique Art Hotel</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer staying in an Eco-Lodge. ")}>Eco-Lodge</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I am looking for a Family-Friendly Stay. ")}>Family-Friendly</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Budget-friendly accommodations. ")}>Budget-Friendly</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Luxury stays. ")}>Luxury</Button>
                      </ButtonGroup>

                      {/* Sub-category 1 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Amenities</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Amenities" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want Free Breakfast included. ")}>Free Breakfast</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want a Pool facility. ")}>Pool</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I need Pet-Friendly options. ")}>Pet-Friendly</Button>
                      </ButtonGroup>

                      {/* Sub-category 2 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Location Preference</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Location Preference" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Downtown areas. ")}>Downtown</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer near Seaside areas. ")}>Seaside</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer quiet, remote locations. ")}>Remote</Button>
                      </ButtonGroup>

                    </Paper>
                  </Box>
                ) : form === "03" ? (
                  <Box sx={{ padding: 2, backgroundColor: '#f0f0f0', borderRadius: 1, minHeight: '200px' }}>
                    {/* Food Preferences */}
                    <Paper className="frosted-card" sx={{ padding: 2 }}>
                      <Typography variant="body1" color="text.primary" mb={1}>Food Preferences</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Food Preferences" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Local Sri Lankan cuisine. ")}>Local Sri Lankan</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer International cuisine. ")}>International</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I am looking for Vegetarian food options. ")}>Vegetarian</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I am interested in Seafood dishes. ")}>Seafood</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Street food experiences. ")}>Street Food</Button>
                      </ButtonGroup>

                      {/* Sub-category 1 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Food Type</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Food Type" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Seafood. ")}>Seafood</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I enjoy Street Food. ")}>Street Food</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer fine dining experiences. ")}>Fine Dining</Button>
                      </ButtonGroup>

                      {/* Sub-category 2 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Dietary Restrictions</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Dietary Restrictions" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I require Gluten-Free options. ")}>Gluten-Free</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I require Vegan options. ")}>Vegan</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I require Halal options. ")}>Halal</Button>
                      </ButtonGroup>

                    </Paper>
                  </Box>
                ) : form === "04" ? (
                  <Box sx={{ padding: 2, backgroundColor: '#f0f0f0', borderRadius: 1, minHeight: '200px' }}>
                    <Paper className="frosted-card" sx={{ padding: 2 }}>
                      <Typography variant="body1" color="text.primary" mb={1}>Activity Preferences</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Activity Preferences" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I am interested in a Wildlife Safari. ")}>Wildlife Safari</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Cultural Heritage tours. ")}>Cultural Heritage</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to explore Food & Culinary tours. ")}>Food & Culinary Tours</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I am interested in Hiking & Nature Trails. ")}>Hiking & Nature Trails</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to enjoy Water Sports. ")}>Water Sports</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Wellness activities like Yoga & Spa. ")}>Wellness</Button>
                      </ButtonGroup>

                      {/* Sub-category 1 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Adventure Activities</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Adventure Activities" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to enjoy Water Sports. ")}>Water Sports</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to try Surfing. ")}>Surfing</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to go Ziplining. ")}>Ziplining</Button>
                      </ButtonGroup>

                      {/* Sub-category 2 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Wellness Activities</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Wellness Activities" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Yoga sessions. ")}>Yoga</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Spa and Massage. ")}>Spa</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Meditation retreats. ")}>Meditation</Button>
                      </ButtonGroup>

                    </Paper>
                  </Box>

                ) : form === "05" ? (
                  <Box sx={{ padding: 2, backgroundColor: '#f0f0f0', borderRadius: 1, minHeight: '200px' }}>
                    {/* Weather Preferences */}
                    <Paper className="frosted-card" sx={{ padding: 2 }}>
                      <Typography variant="body1" color="text.primary" mb={1}>Weather Preferences</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Weather Preferences" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Sunny & Warm weather. ")}>Sunny & Warm</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Mild & Cool weather. ")}>Mild & Cool</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I want to avoid Rainy Areas. ")}>Avoid Rainy Areas</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer cloudy but dry weather. ")}>Cloudy & Dry</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I like cooler temperatures. ")}>Cooler Temperature</Button>
                      </ButtonGroup>

                      {/* Sub-category 1 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Cloud Conditions</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Cloud Conditions" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Cloudy but Dry weather. ")}>Cloudy & Dry</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I like Partly Cloudy weather. ")}>Partly Cloudy</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I prefer Overcast conditions. ")}>Overcast</Button>
                      </ButtonGroup>

                      {/* Sub-category 2 */}
                      <Typography variant="body1" color="text.primary" mb={1} mt={2}>Temperature Preference</Typography>
                      <ButtonGroup variant="contained" color="secondary" aria-label="Temperature Preference" mb={2}>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I like Cooler temperatures. ")}>Cooler Temperature</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I like Warmer temperatures. ")}>Warmer Temperature</Button>
                        <Button sx={{ backgroundColor: buttonColor, color: 'white', '&:hover': { backgroundColor: buttonHover } }} onClick={() => ExtentQuestion("I like Moderate temperatures. ")}>Moderate Temperature</Button>
                      </ButtonGroup>

                    </Paper>
                  </Box>
                ) : null
              )}
            </Box>
          </>
        )}
      </>
    </Container>
  );
};

export default Home;
