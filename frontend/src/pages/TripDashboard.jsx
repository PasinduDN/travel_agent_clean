// src/pages/TripDashboard.jsx

import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
    Container,
    Button,
    Typography,
    Paper,
    Box,
    Grid,
    TextField,
    IconButton,
    Avatar,
    List,
    ListItem,
    ListItemText,
    Card,
    CardContent,
    Link,
    Divider
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import PersonIcon from '@mui/icons-material/Person';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import FacebookIcon from '@mui/icons-material/Facebook';
import TwitterIcon from '@mui/icons-material/Twitter';
import InstagramIcon from '@mui/icons-material/Instagram';
import YouTubeIcon from '@mui/icons-material/YouTube';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const TripDashboard = () => {

    const navigate = useNavigate();
    const location = useLocation();

    const tripPlan = location.state?.tripData || "No trip plan was generated. Go back to create one!";
    const initialAIMessage = location.state?.tripData || "No trip plan was generated. Go back to create one!";

    const [messages, setMessages] = useState([
        { id: 1, sender: 'AI', text: initialAIMessage }
    ]);
    const [newMessage, setNewMessage] = useState('');
    const [socialPosts, setSocialPosts] = useState([]);
    const [likedPosts, setLikedPosts] = useState([]);
    const [previewPost, setPreviewPost] = useState(null);
    const [chatWithAgentOption, setChatWithAgentOption] = useState(location.state.chatWithAgentOption);

    // Social media links and location data
    const [socialLinks] = useState([
        { platform: 'Facebook', url: 'https://facebook.com/srilankatourism', icon: <FacebookIcon /> },
        { platform: 'Twitter', url: 'https://twitter.com/srilankatourism', icon: <TwitterIcon /> },
        { platform: 'Instagram', url: 'https://instagram.com/srilankatourism', icon: <InstagramIcon /> },
        { platform: 'YouTube', url: 'https://youtube.com/srilankatourism', icon: <YouTubeIcon /> }
    ]);

    const [mapLocations] = useState([
        { name: 'Colombo', lat: 6.9271, lng: 79.8612, description: 'Commercial capital of Sri Lanka' },
        { name: 'Kandy', lat: 7.2906, lng: 80.6337, description: 'Cultural capital with Temple of Tooth' },
        { name: 'Galle', lat: 6.0329, lng: 80.2170, description: 'Historic fort city' },
        { name: 'Ella', lat: 6.8667, lng: 81.0500, description: 'Beautiful hill station' }
    ]);

    const handleGoBack = () => {
        navigate('/');
    };

    const handleFirstSendMessage = async () => {
        if (!newMessage.trim()) return; // Don't send empty messages

        const userMessage = {
            id: messages.length + 1,
            sender: 'Human',
            text: newMessage
        };
        setMessages((prev) => [...prev, userMessage]);

        try {
            console.log("handleFirstSendMessage");
            const sessionId = "demo-session"; // or generate dynamically
            const response = await fetch(`http://127.0.0.1:8000/api/process_preferences_forFirstSendMessage/${sessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    preferences_text: newMessage,
                }),
            });
            console.log("response : ", response);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log("data.locations ***** : " + data.locations);

            const aiMessage = {
                id: messages.length + 2, // Use the new length after adding user message
                sender: 'AI',
                text: data.response // Use 'response', not 'result'
            };
            setMessages((prev) => [...prev, aiMessage])

            // setChatWithAgentOption(0)

        } catch (error) {
            console.error("Error fetching AI response:", error);
            console.error("Error fetching AI response:", error);
            // Optional: Show an error message in the chat
            const errorMessage = {
                id: messages.length + 2,
                sender: 'AI',
                text: "Sorry, I couldn't get a response. Please try again."
            };
            setMessages((prev) => [...prev, errorMessage]);
        }

        setNewMessage('');
    }

    const handleSendMessage = async () => {
        console.log("handleSendMessage start");

        if (!newMessage.trim()) return;
        console.log("handleSendMessage end");
        console.log("messages", messages);
        console.log("newMessage", newMessage);
        console.log("selectedLocations", selectedLocations);
        console.log("location.state.start_date", location.state.start_date);
        console.log("location.state.end_date", location.state.end_date);
        console.log("likedPosts", likedPosts);
        console.log("chatWithAgentOption", location.state.chatWithAgentOption);

        const userMessage = {
            id: messages.length + 1,
            sender: 'Human',
            text: newMessage
        };
        setMessages((prev) => [...prev, userMessage]);

        try {
            const sessionId = "demo-session"; // or generate dynamically
            const response = await fetch(`http://127.0.0.1:8000/api/process_preferences/${sessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    preferences_text: newMessage,
                    locations: selectedLocations,
                    start_date: location.state.start_date,
                    end_date: location.state.end_date,
                    liked_posts: likedPosts
                }),
            });

            const data = await response.json();
            console.log("data ***** : " + data);

            if (Array.isArray(data.posts)) {
                setSocialPosts(data.posts);
            }

            // update chat
            const aiMessage = { id: messages.length + 2, sender: 'AI', text: data.result };
            setMessages((prev) => [...prev, aiMessage]);

            // update map markers
            if (Array.isArray(data.locations)) {
                setSelectedLocations(data.locations);
            }
        } catch (error) {
            console.error("Error fetching AI response:", error);
        }

        setNewMessage('');
    };



    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    };

    const [selectedLocations, setSelectedLocations] = useState(location.state?.locations || []);

    const districtCoordinates = {
        Colombo: { lat: 6.9271, lng: 79.8612 },
        Gampaha: { lat: 7.0917, lng: 79.9994 },
        Kalutara: { lat: 6.5854, lng: 80.0940 },
        Kandy: { lat: 7.2906, lng: 80.6337 },
        Matale: { lat: 7.4675, lng: 80.6234 },
        "Nuwara Eliya": { lat: 6.9497, lng: 80.7891 },
        Galle: { lat: 6.0329, lng: 80.2170 },
        Matara: { lat: 5.9549, lng: 80.5540 },
        Hambantota: { lat: 6.1246, lng: 81.1185 },
        Jaffna: { lat: 9.6685, lng: 80.0074 },
        Kilinochchi: { lat: 9.3912, lng: 80.3990 },
        Mannar: { lat: 8.9770, lng: 79.9047 },
        Vavuniya: { lat: 8.7510, lng: 80.4986 },
        Mullaitivu: { lat: 9.2671, lng: 80.8141 },
        Batticaloa: { lat: 7.7316, lng: 81.6747 },
        Ampara: { lat: 7.3018, lng: 81.6747 },
        Trincomalee: { lat: 8.5874, lng: 81.2152 },
        Kurunegala: { lat: 7.4863, lng: 80.3647 },
        Puttalam: { lat: 8.0362, lng: 79.8283 },
        Anuradhapura: { lat: 8.3114, lng: 80.4037 },
        Polonnaruwa: { lat: 7.9403, lng: 81.0188 },
        Badulla: { lat: 6.9896, lng: 81.0560 },
        Monaragala: { lat: 6.8726, lng: 81.3494 },
        Ratnapura: { lat: 6.6828, lng: 80.3994 },
        Kegalle: { lat: 7.2513, lng: 80.3464 },
    };


    return (
        <Container maxWidth="xl" sx={{ paddingY: 4 }}>
            {/* // <Container maxWidth="xl" sx={{ height: '100vh', paddingY: 4 }}> */}
            <Box>
                {/* Header */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#2c3e50' }}>
                        Travel Dashboard
                    </Typography>
                    <Button
                        variant="contained"
                        onClick={handleGoBack}
                        sx={{
                            backgroundColor: '#3498db',
                            '&:hover': { backgroundColor: '#2980b9' },
                            borderRadius: 2,
                            px: 3,
                            fontSize: '1rem',  // Adjusted font size
                            padding: '10px 20px', // Adjust padding for better mobile experience
                            '@media (max-width: 600px)': {
                                fontSize: '0.9rem', // Smaller font size on mobile
                                padding: '8px 16px', // Adjust padding on smaller screens
                            },
                        }}
                    >
                        ← Plan Another Trip
                    </Button>
                </Box>

                <Grid container spacing={2}>
                    {/* Col 01 */}
                    <Grid size={{ xs: 12, md: 6 }} >
                        <Paper
                            elevation={3}
                            sx={{
                                height: '85vh',
                                display: 'flex',
                                flexDirection: 'column',
                                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                borderRadius: 3,
                                overflow: 'hidden'
                            }}
                        >
                            {/* Chat Header */}
                            <Box sx={{
                                padding: 2,
                                backgroundColor: 'rgba(255,255,255,0.1)',
                                borderBottom: '1px solid rgba(255,255,255,0.2)',
                                flexShrink: 0
                            }}>
                                <Typography variant="h6" sx={{ color: 'white', fontWeight: 'bold' }}>
                                    AI Travel Assistant
                                </Typography>
                            </Box>

                            {/* Messages Container */}
                            <Box
                                sx={{
                                    flex: 1,
                                    overflowY: "auto",
                                    padding: 1,
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: 1,
                                    minHeight: 0,
                                }}
                            >
                                {messages.map((message) => (
                                    <Box
                                        key={message.id}
                                        sx={{
                                            display: "flex",
                                            justifyContent:
                                                message.sender === "AI" ? "flex-start" : "flex-end",
                                            mb: 1,
                                        }}
                                    >
                                        <Box
                                            sx={{
                                                display: "flex",
                                                alignItems: "flex-start",
                                                flexDirection:
                                                    message.sender === "AI" ? "row" : "row-reverse",
                                                maxWidth: "80%",
                                            }}
                                        >
                                            <Avatar
                                                sx={{
                                                    width: 32,
                                                    height: 32,
                                                    backgroundColor:
                                                        message.sender === "AI" ? "#4CAF50" : "#2196F3",
                                                    mx: 1,
                                                }}
                                            >
                                                {message.sender === "AI" ? (
                                                    <SmartToyIcon sx={{ fontSize: 18 }} />
                                                ) : (
                                                    <PersonIcon sx={{ fontSize: 18 }} />
                                                )}
                                            </Avatar>
                                            <Paper
                                                sx={{
                                                    padding: 1.5,
                                                    backgroundColor:
                                                        message.sender === "AI"
                                                            ? "rgba(255,255,255,0.9)"
                                                            : "rgba(33,150,243,0.9)",
                                                    color: message.sender === "AI" ? "#333" : "white",
                                                    borderRadius: 2,
                                                    maxWidth: "100%",
                                                }}
                                            >
                                                {message.sender === "AI" ? (
                                                    <ReactMarkdown
                                                        remarkPlugins={[remarkGfm]}
                                                        components={{
                                                            h1: ({ node, ...props }) => (
                                                                <Typography
                                                                    variant="h5"
                                                                    fontWeight="bold"
                                                                    gutterBottom
                                                                    {...props}
                                                                />
                                                            ),
                                                            h2: ({ node, ...props }) => (
                                                                <Typography
                                                                    variant="h6"
                                                                    fontWeight="bold"
                                                                    gutterBottom
                                                                    {...props}
                                                                />
                                                            ),
                                                            strong: ({ node, ...props }) => (
                                                                <Typography component="span" fontWeight="bold" {...props} />
                                                            ),
                                                            p: ({ node, ...props }) => (
                                                                <Typography
                                                                    variant="body2"
                                                                    paragraph
                                                                    sx={{ whiteSpace: "pre-line" }}
                                                                    {...props}
                                                                />
                                                            ),
                                                            li: ({ node, ...props }) => (
                                                                <li style={{ marginLeft: "1.2em" }} {...props} />
                                                            ),
                                                        }}
                                                    >
                                                        {message.text}
                                                    </ReactMarkdown>
                                                ) : (
                                                    <Typography
                                                        variant="body2"
                                                        sx={{ wordWrap: "break-word", whiteSpace: "pre-line" }}
                                                    >
                                                        {message.text}
                                                    </Typography>
                                                )}
                                            </Paper>
                                        </Box>
                                    </Box>
                                ))}
                            </Box>

                            {/* Chat Input */}
                            <Box sx={{
                                padding: 2,
                                backgroundColor: 'rgba(255,255,255,0.1)',
                                borderTop: '1px solid rgba(255,255,255,0.2)',
                                flexShrink: 0
                            }}>
                                <Box sx={{ display: 'flex', gap: 1 }}>
                                    <TextField
                                        fullWidth
                                        size="small"
                                        placeholder="Type your message..."
                                        value={newMessage}
                                        onChange={(e) => setNewMessage(e.target.value)}
                                        onKeyPress={handleKeyPress}
                                        sx={{
                                            '& .MuiOutlinedInput-root': {
                                                backgroundColor: 'rgba(255,255,255,0.9)',
                                                borderRadius: 2,
                                                '& fieldset': { border: 'none' }
                                            }
                                        }}
                                    />
                                    <IconButton
                                        // onClick={chatWithAgentOption === 1 ? handleFirstSendMessage : handleSendMessage}
                                        onClick={handleSendMessage}
                                        sx={{
                                            backgroundColor: '#4CAF50',
                                            color: 'white',
                                            '&:hover': { backgroundColor: '#45a049' },
                                            borderRadius: 2
                                        }}
                                    >
                                        <SendIcon />
                                    </IconButton>
                                </Box>
                            </Box>
                        </Paper>
                    </Grid>

                    {/* Col 02 */}
                    <Grid size={{ xs: 12, md: 2 }}>
                        <Box
                            sx={{
                                display: 'flex',
                                flexDirection: 'column',
                                height: '85vh',
                                gap: 2,
                                overflow: 'hidden',
                            }}
                        >
                            <Typography variant="h6" sx={{ fontWeight: 'bold', color: '#2c3e50', mb: 1 }}>
                                📲 Social Media
                            </Typography>
                            <Typography variant="h7" sx={{ color: '#395a7cff', mb: 1 }}>
                                currently under development
                            </Typography>

                            <Box
                                sx={{
                                    flex: 1,
                                    overflowY: 'auto',
                                    scrollBehavior: 'smooth',
                                    animation: 'scroll-up 60s linear infinite', // auto-scroll
                                }}
                            >
                                {Array.isArray(socialPosts) && socialPosts.length > 0 ? (
                                    socialPosts.map((post, index) => (
                                        <Paper
                                            key={`${post.id}-${index}`}  // Ensure unique key by combining id and index
                                            sx={{
                                                mb: 2,
                                                borderRadius: 2,
                                                overflow: 'hidden',
                                                cursor: 'pointer',
                                            }}
                                            onClick={() => setPreviewPost(post)} // Handle modal preview on click
                                        >
                                            <img
                                                src={post.imageUrl}
                                                alt=""
                                                style={{ width: '100%', height: 'auto' }}
                                            />
                                            <Box sx={{ p: 1 }}>
                                                <Typography variant="subtitle2">@{post.username}</Typography>
                                                <Typography variant="body2">{post.caption}</Typography>
                                                <Typography variant="caption">
                                                    <a href={post.permalink} target="_blank" rel="noopener noreferrer">
                                                        View on Instagram
                                                    </a>
                                                </Typography>

                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                                                    <Button
                                                        size="small"
                                                        onClick={(e) => {
                                                            e.stopPropagation(); // Prevent triggering onClick on Paper
                                                            setLikedPosts((prev) =>
                                                                prev.includes(post.id)
                                                                    ? prev.filter((id) => id !== post.id)
                                                                    : [...prev, post.id]
                                                            );
                                                        }}
                                                    >
                                                        {likedPosts.includes(post.id) ? "❤️ Liked" : "🤍 Like"}
                                                    </Button>
                                                    <Typography variant="caption">{post.likes} likes</Typography>
                                                </Box>
                                            </Box>
                                        </Paper>
                                    ))
                                ) : (
                                    <Typography variant="h6">No posts to show</Typography>
                                )}
                            </Box>

                            {/* Preview Modal */}
                            {previewPost && (
                                <Box
                                    sx={{
                                        position: 'fixed',
                                        top: 0,
                                        left: 0,
                                        width: '100%',
                                        height: '100%',
                                        backgroundColor: 'rgba(0,0,0,0.7)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        zIndex: 2000,
                                    }}
                                >
                                    <Paper sx={{ position: 'relative', p: 2 }}>
                                        <Button
                                            onClick={() => setPreviewPost(null)}
                                            sx={{ position: 'absolute', top: 8, right: 8 }}
                                        >
                                            ✖
                                        </Button>
                                        <img
                                            src={previewPost.imageUrl}
                                            alt=""
                                            style={{ maxWidth: '80vw', maxHeight: '80vh' }}
                                        />
                                        <Typography variant="h6">@{previewPost.username}</Typography>
                                        <Typography>{previewPost.caption}</Typography>
                                        <Typography variant="body2">{previewPost.hashtags.join(" ")}</Typography>
                                    </Paper>
                                </Box>
                            )}
                        </Box>
                    </Grid>



                    {/* Col 03 */}
                    <Grid size={{ xs: 12, md: 4 }}>
                        <Paper
                            elevation={3}
                            sx={{
                                height: '85vh',
                                borderRadius: 3,
                                overflow: 'hidden',
                                background: 'linear-gradient(135deg, #d299c2 0%, #fef9d7 100%)',
                                display: 'flex',
                                flexDirection: 'column'
                            }}
                        >
                            {/* Map Header */}
                            <Box
                                sx={{
                                    padding: 2,
                                    backgroundColor: 'rgba(44,62,80,0.9)',
                                    color: 'white',
                                    flexShrink: 0
                                }}
                            >
                                <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                                    📍 Travel Locations
                                </Typography>
                            </Box>

                            {/* Map Container (fills remaining height) */}
                            <Box sx={{ flex: 1 }}>
                                <MapContainer center={[7.8731, 80.7718]} zoom={7} style={{ height: '100%', width: '100%' }}>
                                    <TileLayer
                                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                        attribution="&copy; OpenStreetMap contributors"
                                    />
                                    {selectedLocations.map((loc) => {
                                        const coords = districtCoordinates[loc];
                                        if (!coords) return null;
                                        return (
                                            <Marker key={loc} position={[coords.lat, coords.lng]}>
                                                <Popup>{loc}</Popup>
                                            </Marker>
                                        );
                                    })}
                                </MapContainer>
                            </Box>
                        </Paper>
                    </Grid>

                </Grid>
            </Box>
        </Container>
    );
};

export default TripDashboard;