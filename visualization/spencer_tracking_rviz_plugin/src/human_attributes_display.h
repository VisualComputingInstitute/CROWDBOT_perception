#ifndef HUMAN_ATTRIBUTES_DISPLAY_H
#define HUMAN_ATTRIBUTES_DISPLAY_H

#include <map>
#include <boost/circular_buffer.hpp>

#include <spencer_human_attribute_msgs/HumanAttributes.h>

#include "person_display_common.h"
#include "tracked_persons_cache.h"
#include "visuals/mesh_node.h"


namespace spencer_tracking_rviz_plugin
{
    /// The display which can be added in RViz to display human attributes.
    class HumanAttributesDisplay: public PersonDisplayCommon<spencer_human_attribute_msgs::HumanAttributes>
    {
    Q_OBJECT
    public:
        // Constructor.  pluginlib::ClassLoader creates instances by calling
        // the default constructor, so make sure you have one.
        HumanAttributesDisplay() {};
        virtual ~HumanAttributesDisplay();

        // Overrides of protected virtual functions from Display.  As much
        // as possible, when Displays are not enabled, they should not be
        // subscribed to incoming data and should not show anything in the
        // 3D view.  These functions are where these connections are made
        // and broken.

        // Called after the constructors have run
        virtual void onInitialize();

        // Called periodically by the visualization manager
        virtual void update(float wall_dt, float ros_dt);

    protected:
        // A helper to clear this display back to the initial state.
        virtual void reset();

        // Must be implemented by derived classes because MOC doesn't work in templates
        virtual rviz::DisplayContext* getContext() {
            return context_;
        }

    private:
        struct HumanAttributeVisual {
            shared_ptr<Ogre::SceneNode> sceneNode;
            shared_ptr<MeshNode> genderMesh;
            unsigned int trackId;
            shared_ptr<TextNode> ageGroupText;
            shared_ptr<TextNode> personHeightText;
        };

        // Functions to handle an incoming ROS message.
        void processMessage(const spencer_human_attribute_msgs::HumanAttributes::ConstPtr& msg);
       
        // Helper functions
        void updateVisualStyles(shared_ptr<HumanAttributeVisual>& humanAttributeVisual);
        shared_ptr<HumanAttributeVisual> createVisualIfNotExists(track_id trackId);

        // User-editable property variables.
        rviz::BoolProperty* m_render_gender_property;
        rviz::BoolProperty* m_render_person_height_property;
        rviz::BoolProperty* m_render_age_group_property;
       
        rviz::FloatProperty* m_occlusion_alpha_property;        

        // State variables
        map<track_id, shared_ptr<HumanAttributeVisual> > m_humanAttributeVisuals;

        Ogre::Matrix4 m_frameTransform;
        TrackedPersonsCache m_trackedPersonsCache;

    private Q_SLOTS:
        virtual void stylesChanged();
    };

} // end namespace spencer_tracking_rviz_plugin

#endif // HUMAN_ATTRIBUTES_DISPLAY_H
